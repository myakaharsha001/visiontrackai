import unittest

from config import ConfigError, DetectionSettings, build_config

GOOD = {"DB_HOST": "localhost", "DB_PORT": "3306", "DB_NAME": "vision_platform",
        "DB_USER": "vision_user", "DB_PASSWORD": "s3cret!"}


class ConfigTests(unittest.TestCase):
    def test_valid_config_uses_defaults(self):
        cfg = build_config(GOOD)
        self.assertEqual(cfg.database.port, 3306)
        self.assertEqual(cfg.model_name, "yolov8n.pt")
        self.assertEqual(cfg.default_confidence, 0.5)

    def test_missing_variables_are_all_listed(self):
        with self.assertRaises(ConfigError) as ctx:
            build_config({"DB_HOST": "localhost"})
        msg = str(ctx.exception)
        for key in ("DB_NAME", "DB_USER", "DB_PASSWORD"):
            self.assertIn(key, msg)
        self.assertIn(".env.example", msg)

    def test_placeholder_password_rejected(self):
        env = dict(GOOD, DB_PASSWORD="replace_with_your_local_password")
        with self.assertRaises(ConfigError) as ctx:
            build_config(env)
        self.assertIn("placeholder", str(ctx.exception))

    def test_empty_password_is_allowed_if_variable_present(self):
        cfg = build_config(dict(GOOD, DB_PASSWORD=""))
        self.assertEqual(cfg.database.password, "")

    def test_invalid_port_and_confidence(self):
        with self.assertRaises(ConfigError) as ctx:
            build_config(dict(GOOD, DB_PORT="abc", DEFAULT_CONFIDENCE="2"))
        msg = str(ctx.exception)
        self.assertIn("DB_PORT", msg)
        self.assertIn("DEFAULT_CONFIDENCE", msg)

    def test_bad_model_and_log_level(self):
        with self.assertRaises(ConfigError):
            build_config(dict(GOOD, MODEL_NAME="model.onnx"))
        with self.assertRaises(ConfigError):
            build_config(dict(GOOD, LOG_LEVEL="LOUD"))

    def test_password_not_in_repr(self):
        self.assertNotIn("s3cret", repr(build_config(GOOD)))

    def test_detection_settings_validation(self):
        DetectionSettings().validate()
        with self.assertRaises(ValueError):
            DetectionSettings(confidence=0.05).validate()
        with self.assertRaises(ValueError):
            DetectionSettings(confidence=0.99).validate()
        with self.assertRaises(ValueError):
            DetectionSettings(process_every_n=0).validate()


if __name__ == "__main__":
    unittest.main()
