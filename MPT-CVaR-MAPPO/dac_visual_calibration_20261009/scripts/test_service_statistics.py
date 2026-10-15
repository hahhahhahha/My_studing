import unittest

from service_summary import quantile, statistics


class ServiceStatisticsTests(unittest.TestCase):
    def test_quantile_is_frozen_order_statistic(self):
        self.assertEqual(quantile([4, 1, 3, 2], .5), 2)
        self.assertEqual(quantile(list(range(1, 101)), .99), 99)

    def test_population_standard_deviation(self):
        result = statistics([1, 3])
        self.assertEqual(result["mean"], 2)
        self.assertEqual(result["population_std"], 1)

    def test_no_tail_trimming(self):
        result = statistics([1] * 98 + [10, 100])
        self.assertEqual(result["n"], 100)
        self.assertEqual(result["P99"], 10)
        self.assertEqual(result["maximum"], 100)


if __name__ == "__main__":
    unittest.main()
