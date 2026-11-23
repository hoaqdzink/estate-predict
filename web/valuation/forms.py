from django import forms

from . import services


class HouseForm(forms.Form):
    """Thông số căn nhà; kiểm tra miền giá trị và quận thuộc thành phố."""
    city = forms.ChoiceField(label="Thành phố", choices=[(c, c) for c in services.CITIES])
    district = forms.CharField(label="Quận/huyện", widget=forms.Select())
    house_type = forms.ChoiceField(label="Loại nhà", choices=[(t, t) for t in services.HOUSE_TYPES],
                                   widget=forms.RadioSelect)
    area_m2 = forms.FloatField(label="Diện tích (m²)", min_value=10, max_value=1000, initial=50)
    road_width_m = forms.FloatField(label="Đường trước nhà (m)", min_value=0.5, max_value=60, initial=4)
    floors = forms.IntegerField(label="Số tầng", min_value=1, max_value=15, initial=4)
    width_m = forms.FloatField(label="Chiều ngang (m)", min_value=1, max_value=50, required=False,
                               help_text="Không bắt buộc")
    has_parking = forms.BooleanField(label="Có chỗ đỗ ô tô", required=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():
            if name in ("has_parking", "house_type"):
                field.widget.attrs["class"] = "form-check-input"
                if name == "has_parking":
                    field.widget.attrs["role"] = "switch"
            elif name in ("city", "district"):
                field.widget.attrs["class"] = "form-select"
            else:
                field.widget.attrs["class"] = "form-control"

    def clean(self):
        data = super().clean()
        city, district = data.get("city"), data.get("district")
        if city and district and district not in services.districts(city):
            self.add_error("district", f"Quận/huyện không thuộc {city}.")
        return data
