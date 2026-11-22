use super::super::ctx::ActionCtx;
use rand::RngExt;

pub fn apply(ctx: &mut ActionCtx) -> f32 {
    let lid = ctx.lid.clone();
    let pick = ctx.near.iter().copied().find(|&k| {
        let o = &ctx.sim.organisms[k];
        o.lineage_id != lid && ctx.sim.organisms[ctx.idx].attitude_toward(&o.lineage_id) < -0.20
    });
    let Some(ki) = pick else {
        ctx.think("no rival in sight");
        return 0.0;
    };
    // Neural hit roll (pwb's acc head): the agile rival slips the first
    // blow more often than not; the duel only lands through them.
    let f = crate::sim::civ::warfare::neuro::acc_features(
        &ctx.sim.organisms[ctx.idx],
        &ctx.sim.organisms[ki],
        &ctx.sim.grid,
    );
    let p = crate::sim::civ::warfare::neuro::neuro_acc(&f);
    if ctx.sim.rng.random::<f32>() >= p {
        let their = ctx.sim.organisms[ki].lineage_id;
        ctx.sim.organisms[ctx.idx].update_attitude(&their, -0.05);
        ctx.think("swung and missed");
        return 0.003;
    }
    {
        let o = &mut ctx.sim.organisms[ki];
        o.health = (o.health - 0.08).max(0.0);
        o.mark_harm(crate::organism::organism::Harm::Fight, ctx.tick);
    }
    {
        let me = &mut ctx.sim.organisms[ctx.idx];
        me.health = (me.health - 0.04).max(0.0);
        me.mark_harm(crate::organism::organism::Harm::Fight, ctx.tick);
    }
    let their = ctx.sim.organisms[ki].lineage_id;
    ctx.sim.organisms[ctx.idx].update_attitude(&their, -0.05);
    ctx.think("duelling a rival");
    ctx.discover("duelling", "duelled a rival");
    0.006
}
