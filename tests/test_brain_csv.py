import os
import tempfile
import unittest
import warnings

from src.brain import migrate_results_csv_context, read_simulations_csv, save_alpha_to_csv


class TestBrainCsv(unittest.TestCase):
    def test_read_headered_simulation_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "simulation_results.csv")
            with open(csv_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "alpha_id,regular_code,turnover,returns,drawdown,margin,fitness,sharpe,"
                    "LOW_SHARPE,LOW_FITNESS,LOW_TURNOVER,HIGH_TURNOVER,CONCENTRATED_WEIGHT,"
                    "LOW_SUB_UNIVERSE_SHARPE,SELF_CORRELATION,MATCHES_COMPETITION,date,"
                    "region,universe,delay,decay,neutralization,truncation\n"
                )
                handle.write(
                    "A1,rank(close),0.2,0.1,0.05,0.03,1.4,1.8,"
                    "PASS,PASS,PASS,PASS,PASS,PASS,PASS,PASS,2026-03-25 10:00:00,"
                    "USA,TOP3000,1,5,Industry,0.05\n"
                )

            df = read_simulations_csv(csv_path)

            self.assertEqual(len(df), 1)
            self.assertEqual(df.loc[0, "alpha_id"], "A1")
            self.assertEqual(df.loc[0, "regular_code"], "rank(close)")
            self.assertIn("date", df.columns)
            self.assertEqual(df.loc[0, "region"], "USA")
            self.assertEqual(df.loc[0, "universe"], "TOP3000")

    def test_read_legacy_headerless_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "simulations.csv")
            with open(csv_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "A2,rank(vwap),0.1,0.08,0.03,0.02,1.2,1.5,"
                    "PASS,PASS,PASS,PASS,PASS,PASS,FAIL,PASS,2026-03-25 11:00:00\n"
                )

            df = read_simulations_csv(csv_path)

            self.assertEqual(len(df), 1)
            self.assertEqual(df.loc[0, "alpha_id"], "A2")
            self.assertEqual(df.loc[0, "SELF_CORRELATION"], "FAIL")
            self.assertIn("date", df.columns)

    def test_save_alpha_to_csv_writes_standard_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            previous_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                save_alpha_to_csv(
                    {
                        "alpha_id": "A3",
                        "regular_code": "rank(ts_zscore(close,21))",
                        "turnover": 0.23,
                        "returns": 0.12,
                        "drawdown": 0.04,
                        "margin": 0.03,
                        "fitness": 1.55,
                        "sharpe": 2.02,
                        "LOW_SHARPE": "PASS",
                        "LOW_FITNESS": "PASS",
                        "LOW_TURNOVER": "PASS",
                        "HIGH_TURNOVER": "PASS",
                        "CONCENTRATED_WEIGHT": "PASS",
                        "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                        "SELF_CORRELATION": "PASS",
                        "MATCHES_COMPETITION": "PASS",
                        "region": "USA",
                        "universe": "TOP3000",
                        "delay": 1,
                        "decay": 5,
                        "neutralization": "Industry",
                        "truncation": 0.05,
                    }
                )

                df = read_simulations_csv("simulation_results.csv")
                self.assertEqual(len(df), 1)
                self.assertEqual(df.loc[0, "alpha_id"], "A3")
                self.assertEqual(df.loc[0, "SELF_CORRELATION"], "PASS")
                self.assertEqual(df.loc[0, "region"], "USA")
                self.assertEqual(df.loc[0, "delay"], 1)
            finally:
                os.chdir(previous_cwd)

    def test_save_alpha_to_csv_replaces_pending_placeholder_when_final_alpha_arrives(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            previous_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                save_alpha_to_csv(
                    {
                        "alpha_id": "",
                        "regular_code": "rank(close)",
                        "turnover": None,
                        "returns": None,
                        "drawdown": None,
                        "margin": None,
                        "fitness": None,
                        "sharpe": None,
                        "LOW_SHARPE": "PENDING",
                        "LOW_FITNESS": "PENDING",
                        "LOW_TURNOVER": "PENDING",
                        "HIGH_TURNOVER": "PENDING",
                        "CONCENTRATED_WEIGHT": "PENDING",
                        "LOW_SUB_UNIVERSE_SHARPE": "PENDING",
                        "SELF_CORRELATION": "PENDING",
                        "MATCHES_COMPETITION": "PENDING",
                        "region": "USA",
                        "universe": "TOP3000",
                        "delay": 1,
                        "decay": 5,
                        "neutralization": "Market",
                        "truncation": 0.05,
                    }
                )
                save_alpha_to_csv(
                    {
                        "alpha_id": "A5",
                        "regular_code": "rank(close)",
                        "turnover": 0.2,
                        "returns": 0.1,
                        "drawdown": 0.04,
                        "margin": 0.03,
                        "fitness": 1.4,
                        "sharpe": 1.9,
                        "LOW_SHARPE": "PASS",
                        "LOW_FITNESS": "PASS",
                        "LOW_TURNOVER": "PASS",
                        "HIGH_TURNOVER": "PASS",
                        "CONCENTRATED_WEIGHT": "PASS",
                        "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                        "SELF_CORRELATION": "PASS",
                        "MATCHES_COMPETITION": "PASS",
                        "region": "USA",
                        "universe": "TOP3000",
                        "delay": 1,
                        "decay": 5,
                        "neutralization": "Market",
                        "truncation": 0.05,
                    }
                )

                df = read_simulations_csv("simulation_results.csv")
                self.assertEqual(len(df), 1)
                self.assertEqual(df.loc[0, "alpha_id"], "A5")
                self.assertEqual(df.loc[0, "LOW_SHARPE"], "PASS")
            finally:
                os.chdir(previous_cwd)

    def test_save_alpha_to_csv_does_not_raise_concat_futurewarning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            previous_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                first = {
                    "alpha_id": "A6",
                    "regular_code": "rank(close)",
                    "turnover": None,
                    "returns": None,
                    "drawdown": None,
                    "margin": None,
                    "fitness": None,
                    "sharpe": None,
                    "LOW_SHARPE": "PENDING",
                    "LOW_FITNESS": "PENDING",
                    "LOW_TURNOVER": "PENDING",
                    "HIGH_TURNOVER": "PENDING",
                    "CONCENTRATED_WEIGHT": "PENDING",
                    "LOW_SUB_UNIVERSE_SHARPE": "PENDING",
                    "SELF_CORRELATION": "PENDING",
                    "MATCHES_COMPETITION": "PENDING",
                    "region": "USA",
                    "universe": "TOP3000",
                    "delay": 1,
                    "decay": 5,
                    "neutralization": "Market",
                    "truncation": 0.05,
                }
                second = {
                    "alpha_id": "A7",
                    "regular_code": "rank(vwap)",
                    "turnover": 0.2,
                    "returns": 0.1,
                    "drawdown": 0.04,
                    "margin": 0.03,
                    "fitness": 1.4,
                    "sharpe": 1.9,
                    "LOW_SHARPE": "PASS",
                    "LOW_FITNESS": "PASS",
                    "LOW_TURNOVER": "PASS",
                    "HIGH_TURNOVER": "PASS",
                    "CONCENTRATED_WEIGHT": "PASS",
                    "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                    "SELF_CORRELATION": "PASS",
                    "MATCHES_COMPETITION": "PASS",
                    "region": "USA",
                    "universe": "TOP3000",
                    "delay": 1,
                    "decay": 5,
                    "neutralization": "Market",
                    "truncation": 0.05,
                }

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    save_alpha_to_csv(first)
                    save_alpha_to_csv(second)

                future_warnings = [item for item in caught if issubclass(item.category, FutureWarning)]
                self.assertEqual(future_warnings, [])
            finally:
                os.chdir(previous_cwd)

    def test_migrate_results_csv_context_backfills_legacy_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "simulation_results.csv")
            with open(csv_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "alpha_id,regular_code,turnover,returns,drawdown,margin,fitness,sharpe,"
                    "LOW_SHARPE,LOW_FITNESS,LOW_TURNOVER,HIGH_TURNOVER,CONCENTRATED_WEIGHT,"
                    "LOW_SUB_UNIVERSE_SHARPE,SELF_CORRELATION,MATCHES_COMPETITION,date\n"
                )
                handle.write(
                    "A4,rank(close),0.2,0.1,0.05,0.03,1.4,1.8,"
                    "PASS,PASS,PASS,PASS,PASS,PASS,PASS,PASS,2026-03-25 10:00:00\n"
                )

            migration = migrate_results_csv_context(csv_path)
            df = read_simulations_csv(csv_path)

            self.assertEqual(migration["status"], "migrated")
            self.assertEqual(len(df), 1)
            self.assertEqual(df.loc[0, "region"], "USA")
            self.assertEqual(df.loc[0, "universe"], "TOP3000")
            self.assertEqual(int(df.loc[0, "delay"]), 1)


if __name__ == "__main__":
    unittest.main()
