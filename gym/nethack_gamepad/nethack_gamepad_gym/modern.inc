/* Modern-style revival adapted from GnollHack src/end.c:done(), and
 * src/do.c:heal_ailments_upon_revival(). Recovery now uses native level entries.
 * Original references and GnollHack/NetHack license: external/gnollhack_reference.
 * Modified 2026-11-12: NetHack 3.6.0 statuses, entrance checkpoints, no altar logic.
 * This is included in end.c so it can reuse NetHack's static savelife().
 */

#include <limits.h>

extern void desktop_game_event(const char *, int, const char *);

long
desktop_score_divisor(void)
{
    /* GnollHack end.c Modern mortality multiplier, in integer form.
     * Base divisor is 3, then 9, 27, 54, 81, ... for successive true deaths.
     */
    if (!u.desktop_modern)
        return 1L;
    if (u.desktop_revivals == 0)
        return 3L;
    if (u.desktop_revivals == 1)
        return 9L;
    if (u.desktop_revivals == 2)
        return 27L;
    return (u.desktop_revivals - 1L) > LONG_MAX / 27L
               ? LONG_MAX : 27L * (u.desktop_revivals - 1L);
}

void
desktop_record_entry(const char *reason)
{
    if (!u.desktop_modern || u.desktop_revival_pending)
        return;
    assign_level(&u.desktop_entry_level, &u.uz);
    u.desktop_entry_x = u.ux;
    u.desktop_entry_y = u.uy;
    u.desktop_entry_valid = TRUE;
    ++u.desktop_entry_id;
    Strcpy(u.desktop_entry_reason, reason);
    desktop_game_event("checkpoint", 0, reason);
}

void
desktop_modern_init(void)
{
    const char *mode = getenv("NETHACK_DESKTOP_MODE");

    u.desktop_modern = mode && !strcmp(mode, "modern");
    desktop_record_entry("start");
}

static boolean
desktop_recovery_square(int x, int y)
{
    /* goodpos supplies native occupancy/passability checks. Recovery also
     * excludes traps (including portals), water, lava, closed doors and rocks.
     */
    return isok(x, y) && accessible(x, y) && !closed_door(x, y)
           && !t_at(x, y) && !is_pool(x, y) && !is_lava(x, y)
           && !sobj_at(BOULDER, x, y) && goodpos(x, y, &youmonst, 0);
}

void
desktop_finish_revival(void)
{
    int x, y, recovery_x = 0, recovery_y = 0;
    int distance, best_distance = COLNO + ROWNO;

    /* Keep the recorded entrance fixed; an obstruction changes this landing,
     * not the checkpoint. No altar scan, altar creation or alignment logic.
     */
    for (x = 1; x < COLNO; ++x)
        for (y = 0; y < ROWNO; ++y) {
            distance = abs(x - u.desktop_entry_x) + abs(y - u.desktop_entry_y);
            if (distance < best_distance && desktop_recovery_square(x, y)) {
                recovery_x = x;
                recovery_y = y;
                best_distance = distance;
            }
        }
    if (recovery_x)
        teleds(recovery_x, recovery_y, FALSE);
    else {
        /* Never fall back onto a portal or hazardous square. A pathological
         * floor with no valid landing recovers in place and records the fallback.
         */
        desktop_game_event("recovery_fallback", u.desktop_death_how,
                           "no safe landing on current level");
    }
    u.desktop_revival_pending = FALSE;
    u.desktop_shield_until = moves + 2;
    context.botl = context.botlx = 1;
    docrt();
    desktop_game_event("revival", u.desktop_death_how, u.desktop_death_cause);
    pline("After being dead for a while, you feel the saving grace of %s, and wake up%s.",
          u_gname(), recovery_x ? " near your level entrance" : " where you fell");
    display_nhwindow(WIN_MESSAGE, TRUE);
}

void
desktop_revive(int how)
{
    /* Capture an arrival before healing, if a lethal arrival effect ran before
     * the usual placement hook. Never send a character to a previous floor.
     */
    if (!u.desktop_entry_valid || !on_level(&u.desktop_entry_level, &u.uz))
        desktop_record_entry("arrival fallback");

    ++u.desktop_revivals;
    u.desktop_death_how = how;
    Strcpy(u.desktop_death_cause, killer.name);
    desktop_game_event("death", how, killer.name);
    savelife(how);
    /* NetHack 3.6.0 equivalents of GnollHack's revival ailment cleanup. */
    make_blinded(0L, FALSE);
    make_stunned(0L, FALSE);
    make_confused(0L, FALSE);
    make_deaf(0L, FALSE);
    make_slimed(0L, (const char *) 0);
    make_sick(0L, (const char *) 0, FALSE, SICK_ALL);
    make_stoned(0L, (const char *) 0, 0, (const char *) 0);
    (void) make_hallucinated(0L, FALSE, 0L);
    HConflict &= ~TIMEOUT;
    if (Upolyd)
        u.mh = u.mhmax;
    nomul(0);
    u.uinvulnerable = 0;
    u.desktop_revival_pending = TRUE;
    u.desktop_shield_until = moves + 2;

    desktop_finish_revival();
}
