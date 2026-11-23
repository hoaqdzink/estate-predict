from django import template

register = template.Library()


@register.filter
def vn(value, digits=2):
    """Định dạng số kiểu Việt Nam: 1.234,56"""
    try:
        s = f"{float(value):,.{int(digits)}f}"
    except (TypeError, ValueError):
        return "–"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


@register.filter
def get_item(mapping, key):
    """Lấy giá trị theo khóa có dấu cách hoặc ký tự đặc biệt, ví dụ m|get_item:"MAPE (%)"."""
    return mapping.get(key) if hasattr(mapping, "get") else None
