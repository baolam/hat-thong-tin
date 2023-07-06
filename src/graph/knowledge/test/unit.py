from ..unit.NotificationUnit import NotificationUnit

unit = NotificationUnit("Xin chào", 64)

def show(**kwargs):
    content = kwargs.get("content")
    print("Nội dung được gọi là:", content)

unit._add_call(show)