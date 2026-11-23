import json

from django.test import SimpleTestCase

from . import services

RESULT = 'class="price-card'
HOUSE = {"city": "Hà Nội", "district": "Cầu Giấy", "house_type": "Nhà trong hẻm",
         "area_m2": 50, "road_width_m": 3, "floors": 5}


class PagesTest(SimpleTestCase):
    def test_index_get(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "Thông số căn nhà")
        self.assertContains(r, "Cầu Giấy")

    def test_index_post_valid(self):
        r = self.client.post("/", HOUSE)
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, RESULT)
        self.assertContains(r, "Khoảng dự báo 95%")
        self.assertContains(r, "Vì sao ra mức giá này")

    def test_index_post_district_not_in_city(self):
        r = self.client.post("/", {**HOUSE, "district": "Quận 1"})
        self.assertContains(r, "không thuộc Hà Nội")
        self.assertNotContains(r, RESULT)

    def test_index_post_out_of_range(self):
        r = self.client.post("/", {**HOUSE, "area_m2": 5})
        self.assertNotContains(r, RESULT)

    def test_model_and_api_pages(self):
        self.assertContains(self.client.get("/mo-hinh/"), "Trọng số mô hình meta")
        self.assertContains(self.client.get("/api/"), "/api/predict/")


class ApiTest(SimpleTestCase):
    def post(self, payload, query=""):
        return self.client.post("/api/predict/" + query, json.dumps(payload), content_type="application/json")

    def test_districts(self):
        r = self.client.get("/api/districts/", {"city": "Hồ Chí Minh"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("Quận 1", r.json()["districts"])
        self.assertEqual(self.client.get("/api/districts/", {"city": "Đà Nẵng"}).status_code, 400)

    def test_predict_matches_model(self):
        r = self.post(HOUSE)
        self.assertEqual(r.status_code, 200)
        d = r.json()
        direct = services.predict(**HOUSE, explain=False)["price_billion_vnd"]
        self.assertAlmostEqual(d["price_billion_vnd"], direct, places=6)
        self.assertLess(d["interval_95"]["low"], d["price_billion_vnd"])
        self.assertGreater(d["interval_95"]["high"], d["price_billion_vnd"])
        self.assertEqual(len(d["stacking"]["base_models"]), 5)

    def test_shap_adds_up(self):
        d = self.post(HOUSE).json()
        import numpy as np
        total = np.log1p(d["explanation"]["base_price_billion_vnd"]) + sum(
            c["shap_log1p"] for c in d["explanation"]["contributions"])
        self.assertAlmostEqual(total, d["stacking"]["final_log1p"], places=2)

    def test_predict_without_explanation(self):
        self.assertNotIn("explanation", self.post(HOUSE, "?explain=0").json())

    def test_predict_errors(self):
        self.assertEqual(self.post({**HOUSE, "floors": 0}).status_code, 400)
        self.assertIn("district", self.post({**HOUSE, "district": "Quận 1"}).json()["fields"])
        r = self.client.post("/api/predict/", "không phải json", content_type="application/json")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.client.get("/api/predict/").status_code, 405)
