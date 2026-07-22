import copy
import json
import tempfile
import unittest
from pathlib import Path

from verified_memory.m1_context import (
    CONTEXT_MODES,
    FEATURE_SCHEMA_VERSION,
    PROJECTION_SCHEMA_VERSION,
    CausalContextRouter,
    ContextPacket,
    ContextRoute,
    FrozenLinearProjection,
)


def history(count: int = 4):
    return [
        {
            "timestamp": month,
            "price": 100.0 + month,
            "interest_rate": 0.03 + 0.001 * month,
            "unemployment_rate": 0.05 - 0.002 * month,
            "inflation": 0.01 + 0.005 * month,
            "sentiment": -0.10 + 0.05 * month,
        }
        for month in range(count)
    ]


def training_metadata():
    return {
        "training_run_id": "independent-seeds-v1",
        "training_seeds": [101, 202],
        "target_definition": "next-period macro-state residual",
        "fit_method": "ridge",
        "uses_future_inputs": False,
    }


class CausalContextRouterTest(unittest.TestCase):
    def test_all_four_channel_modes_are_independent(self):
        router = CausalContextRouter(window_size=3)
        packet = router.encode(history(), decision_t=3)
        expected = {
            "no-context": (False, False),
            "prompt-only": (False, True),
            "retrieval-only": (True, False),
            "full": (True, True),
        }

        self.assertEqual(set(expected), set(CONTEXT_MODES))
        for mode, (to_retrieval, to_prompt) in expected.items():
            with self.subTest(mode=mode):
                route = router.route(packet, mode=mode)
                self.assertEqual(route.to_retrieval, to_retrieval)
                self.assertEqual(route.to_prompt, to_prompt)
                self.assertEqual(route.context_id, packet.context_id)
                if to_retrieval:
                    self.assertEqual(route.retrieval_vector, packet.context_vector)
                else:
                    self.assertIsNone(route.retrieval_vector)
                if to_prompt:
                    self.assertEqual(route.prompt_summary, packet.prompt_summary)
                else:
                    self.assertEqual(route.prompt_summary, "")
                self.assertEqual(ContextRoute.from_json(route.to_json()), route)

    def test_rolling_features_use_only_configured_window(self):
        router = CausalContextRouter(
            base_feature_names=("price",), window_size=2
        )
        packet = router.encode(history(), decision_t=3)

        self.assertEqual(packet.history_start, 2)
        self.assertEqual(packet.observation_count, 2)
        self.assertEqual(
            packet.feature_names, ("price.last", "price.mean", "price.slope")
        )
        self.assertEqual(packet.raw_features, (103.0, 102.5, 1.0))

    def test_time_is_monotonic_and_future_inputs_are_rejected(self):
        router = CausalContextRouter()
        with self.assertRaisesRegex(ValueError, "future observation"):
            router.encode(history(5), decision_t=3, observed_through=3)

        out_of_order = history(4)
        out_of_order[2], out_of_order[3] = out_of_order[3], out_of_order[2]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            router.encode(out_of_order, decision_t=3)

        with self.assertRaisesRegex(ValueError, "last history timestamp"):
            router.encode(history(3), decision_t=3, observed_through=3)

        with self.assertRaisesRegex(ValueError, "later than decision_t"):
            router.encode(history(4), decision_t=2, observed_through=3)

    def test_future_event_is_rejected_and_absent_event_is_explicit(self):
        router = CausalContextRouter(
            event_feature_names=("direction", "magnitude")
        )
        without_event = router.encode(history(), decision_t=3)
        self.assertEqual(without_event.raw_features[-3:], (0.0, 0.0, 0.0))

        with self.assertRaisesRegex(ValueError, "event timestamp"):
            router.encode(
                history(),
                decision_t=3,
                event={"timestamp": 4, "direction": -1, "magnitude": 0.5},
            )

        with_event = router.encode(
            history(),
            decision_t=3,
            event={"timestamp": 3, "direction": -1, "magnitude": 0.5},
        )
        self.assertEqual(with_event.raw_features[-3:], (1.0, -1.0, 0.5))
        self.assertIn("event_direction=-1", with_event.prompt_summary)

    def test_context_ids_are_deterministic_and_integrity_checked(self):
        router = CausalContextRouter(window_size=3)
        first = router.encode(history(), decision_t=3)
        reordered = [dict(reversed(list(row.items()))) for row in history()]
        second = router.encode(reordered, decision_t=3)

        self.assertEqual(first.context_id, second.context_id)
        self.assertEqual(first.context_hash, second.context_hash)
        restored = ContextPacket.from_json(first.to_json())
        self.assertEqual(restored, first)

        tampered = first.to_dict()
        tampered["context_vector"][0] += 1.0
        with self.assertRaisesRegex(ValueError, "context_hash"):
            ContextPacket.from_dict(tampered)

    def test_new_causal_observation_changes_id_but_irrelevant_extra_fields_do_not(self):
        router = CausalContextRouter(window_size=3)
        base = history()
        first = router.encode(base, decision_t=3)

        with_extra = copy.deepcopy(base)
        with_extra[-1]["future_target_not_in_schema"] = 999999
        same = router.encode(with_extra, decision_t=3)
        self.assertEqual(first.context_id, same.context_id)

        changed = copy.deepcopy(base)
        changed[-1]["inflation"] += 0.01
        different = router.encode(changed, decision_t=3)
        self.assertNotEqual(first.context_id, different.context_id)

    def test_router_round_trip_preserves_default_mode(self):
        router = CausalContextRouter(
            base_feature_names=("price", "inflation"),
            event_feature_names=("direction",),
            window_size=2,
            mode="prompt_only",
        )
        restored = CausalContextRouter.from_json(router.to_json())
        self.assertEqual(restored.to_dict(), router.to_dict())
        packet = restored.encode(
            history(),
            decision_t=3,
            event={"timestamp": 3, "direction": -1.0},
        )
        route = restored.route(packet)
        self.assertFalse(route.to_retrieval)
        self.assertTrue(route.to_prompt)


class FrozenProjectionTest(unittest.TestCase):
    def projection_document(self, router):
        inputs = list(router.feature_names)
        return {
            "schema_version": PROJECTION_SCHEMA_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "projection_id": "ridge-z-v1",
            "input_features": inputs,
            "output_features": ["z0", "z1"],
            "feature_config": router.feature_config,
            "weights": [
                [1.0] + [0.0] * (len(inputs) - 1),
                [0.0, 1.0] + [0.0] * (len(inputs) - 2),
            ],
            "bias": [0.5, -0.5],
            "training_metadata": training_metadata(),
        }

    def test_frozen_projection_loads_from_json_and_is_applied(self):
        probe = CausalContextRouter(
            base_feature_names=("price",), window_size=2
        )
        document = self.projection_document(probe)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "projection.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            router = CausalContextRouter.with_projection_json(
                path, base_feature_names=("price",), window_size=2
            )

        packet = router.encode(history(), decision_t=3)
        self.assertEqual(packet.raw_features, (103.0, 102.5, 1.0))
        self.assertEqual(packet.vector_feature_names, ("z0", "z1"))
        self.assertEqual(packet.context_vector, (103.5, 102.0))
        self.assertEqual(packet.encoder_version, "ridge-z-v1")
        self.assertEqual(
            packet.projection_metadata["training_metadata"], training_metadata()
        )

        restored = CausalContextRouter.from_json(router.to_json())
        restored_packet = restored.encode(history(), decision_t=3)
        self.assertEqual(restored_packet, packet)

    def test_projection_feature_order_and_versions_are_strict(self):
        router = CausalContextRouter(base_feature_names=("price",))
        document = self.projection_document(router)

        wrong_order = copy.deepcopy(document)
        wrong_order["input_features"] = list(reversed(wrong_order["input_features"]))
        with self.assertRaisesRegex(ValueError, "schema/order mismatch"):
            FrozenLinearProjection.from_dict(
                wrong_order, expected_input_features=router.feature_names
            )

        wrong_version = copy.deepcopy(document)
        wrong_version["schema_version"] = "future-version"
        with self.assertRaisesRegex(ValueError, "unsupported projection"):
            FrozenLinearProjection.from_dict(wrong_version)

        extra_key = copy.deepcopy(document)
        extra_key["weigths"] = extra_key["weights"]
        with self.assertRaisesRegex(ValueError, "extra"):
            FrozenLinearProjection.from_dict(extra_key)

        wrong_window = copy.deepcopy(document)
        wrong_window["feature_config"]["window_size"] += 1
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            FrozenLinearProjection.from_dict(
                wrong_window,
                expected_input_features=router.feature_names,
                expected_feature_config=router.feature_config,
            )

    def test_projection_rejects_missing_or_future_input_training_metadata(self):
        router = CausalContextRouter(base_feature_names=("price",))
        document = self.projection_document(router)
        del document["training_metadata"]["training_seeds"]
        with self.assertRaisesRegex(ValueError, "training_seeds"):
            FrozenLinearProjection.from_dict(document)

        future = self.projection_document(router)
        future["training_metadata"]["uses_future_inputs"] = True
        with self.assertRaisesRegex(ValueError, "uses_future_inputs=false"):
            FrozenLinearProjection.from_dict(future)


if __name__ == "__main__":
    unittest.main()
