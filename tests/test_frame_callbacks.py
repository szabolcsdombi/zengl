import zengl


def test_callbacks(ctx: zengl.Context):
    lst = []

    def before_frame():
        lst.append("before_frame")

    def after_frame():
        lst.append("after_frame")

    ctx.before_frame = before_frame
    ctx.after_frame = after_frame

    assert lst == []

    ctx.new_frame()
    ctx.end_frame()

    assert lst == ["before_frame", "after_frame"]

    ctx.new_frame()
    ctx.end_frame()

    assert lst == ["before_frame", "after_frame"] * 2

    ctx.before_frame = None
    ctx.after_frame = None

    ctx.new_frame()
    ctx.end_frame()

    assert lst == ["before_frame", "after_frame"] * 2
