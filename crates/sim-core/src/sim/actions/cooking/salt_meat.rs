use super::super::ctx::ActionCtx;
pub fn apply(ctx: &mut ActionCtx) -> f32 {
    if !ctx.water_near || ctx.org().inv_food == 0 {
        return 0.0;
    }
    let o = ctx.org_mut();
    let cur = o.tools.get("preserved").copied().unwrap_or(0);
    // Saturate: a person who salts meat all their life must not overflow the
    // u8 tool counter in debug builds (found by the long dispersal test).
    o.tools.insert("preserved".to_string(), cur.saturating_add(1));
    ctx.think("salting meat");
    ctx.discover("salt-curing", "learned to salt meat");
    0.008
}
