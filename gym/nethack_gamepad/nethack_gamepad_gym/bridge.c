/* Desktop transport for NetHack's native window_procs.
 * Modified 2026-11-12: optional Modern rules in modern.inc and structured events.
 * NetHack may be freely redistributed. See external/nethack_ds/dat/license.
 */
#include "hack.h"
#include "dlb.h"
#include "func_tab.h"

extern short glyph2tile[];
extern void substitute_tiles(d_level *);
static int window_type[128], item_count[128], next_window = 1;
static anything identifiers[128][1024];
static char response[16384];
static char *messages[100];
static int message_count = 0, history_index = 0;

static void remember(const char *text) {
    if (!text) return;
    if (message_count == 100) {
        free(messages[0]);
        memmove(messages, messages + 1, 99 * sizeof(char *));
        --message_count;
    }
    messages[message_count++] = dupstr(text);
}
static void bridge_end_menu(winid, const char *);
#define ISWHITESPACE(c) ((c) == ' ' || (c) == '\t' || (c) == '\r' || (c) == '\n')
#include "nds_choices.inc"

static void
quoted(const char *text)
{
    const unsigned char *p = (const unsigned char *) (text ? text : "");
    putchar('"');
    for (; *p; ++p) {
        if (*p == '"' || *p == '\\') printf("\\%c", *p);
        else if (*p < 32 || *p > 126) printf("\\u%04x", *p);
        else putchar(*p);
    }
    putchar('"');
}

static void
read_response(void)
{
    fflush(stdout);
    if (!fgets(response, sizeof response, stdin)) {
        program_state.done_hup = 1;
        end_of_input();
    }
}

static void
bridge_checkpoint(void)
{
    if (!u.desktop_entry_valid) {
        printf("null");
        return;
    }
    printf("{\"id\":%ld,\"dungeon\":%d,\"level\":%d,\"x\":%d,\"y\":%d,\"reason\":",
           u.desktop_entry_id, u.desktop_entry_level.dnum, u.desktop_entry_level.dlevel,
           u.desktop_entry_x, u.desktop_entry_y);
    quoted(u.desktop_entry_reason);
    printf("}");
}

void
desktop_game_event(const char *kind, int how, const char *cause)
{
    printf("{\"event\":"); quoted(kind);
    printf(",\"cause\":"); quoted(cause);
    printf(",\"how\":%d,\"turn\":%ld,\"dungeon\":%d,\"level\":%d,"
           "\"x\":%d,\"y\":%d,\"hp\":%d,\"revivals\":%ld,\"checkpoint\":",
           how, moves, u.uz.dnum, u.uz.dlevel, u.ux, u.uy, u.uhp, u.desktop_revivals);
    bridge_checkpoint();
    printf("}\n");
}

static void
bridge_mode(void)
{
    printf("{\"event\":\"mode\",\"mode\":\"%s\",\"revivals\":%ld,\"score_divisor\":%ld,\"checkpoint\":",
           u.desktop_modern ? "modern" : "classic", u.desktop_revivals, desktop_score_divisor());
    bridge_checkpoint();
    printf("}\n");
}

static void
input_request(const char *kind, const char *prompt, const char *choices, int def)
{
    bridge_mode();
    printf("{\"event\":\"input\",\"kind\":"); quoted(kind);
    printf(",\"prompt\":"); quoted(prompt);
    printf(",\"choices\":"); quoted(choices);
    printf(",\"default\":%d,\"turn\":%ld,\"hp\":%d,\"maxhp\":%d,"
           "\"x\":%d,\"y\":%d}\n", def, moves, u.uhp, u.uhpmax, u.ux, u.uy);
    read_response();
}

static void
bridge_init(int *argc, char **argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    iflags.window_inited = TRUE;
    printf("{\"event\":\"hello\",\"engine\":\"NetHack 3.6.0\","
           "\"protocol\":1}\n");
}

boolean desktop_targeting = FALSE;

static int
bridge_getch(void)
{
    input_request(desktop_targeting ? "position" : "key", "", "", 0);
#ifdef DESKTOP_TEST_DRIVER
    /* Compiled only by --test-driver, isolated from participant runtimes. */
    if (atoi(response) >= 250 && atoi(response) <= 256 && atoi(response) != 251) {
        int how = atoi(response) == 252 ? STONING
                  : atoi(response) == 253 ? DROWNING
                  : atoi(response) == 254 ? BURNING
                  : atoi(response) == 256 ? GENOCIDED : DIED;
        if (atoi(response) == 255) {
            make_sick(3L, "desktop test sickness", FALSE, SICK_ALL);
            make_slimed(2L, "desktop test slime");
            make_blinded(20L, FALSE);
        }
        Strcpy(killer.name, "desktop test hazard");
        u.uhp = 0;
        done(how);
        return '\033';
    }
    if (atoi(response) == 258 || atoi(response) == 259 || atoi(response) == 266) {
        d_level destination;
        destination.dnum = 0;
        destination.dlevel = atoi(response) == 258 ? 1 : atoi(response) == 259 ? 3 : 2;
        goto_level(&destination, atoi(response) != 266, atoi(response) == 266, FALSE);
        return '\033';
    }
    if (atoi(response) == 260) {
        (void) safe_teleds(FALSE);
        return '\033';
    }
    if (atoi(response) == 261 || atoi(response) == 262 || atoi(response) == 263) {
        int command = atoi(response), tries = 0;
        while (u.ux == u.desktop_entry_x && u.uy == u.desktop_entry_y && tries++ < 20)
            (void) safe_teleds(FALSE);
        if (command == 261) {
            levl[u.desktop_entry_x][u.desktop_entry_y].typ = ROOM;
            if (!maketrap(u.desktop_entry_x, u.desktop_entry_y, MAGIC_PORTAL))
                panic("test portal placement failed");
        } else if (command == 262)
            levl[u.desktop_entry_x][u.desktop_entry_y].typ = POOL;
        else
            (void) makemon(&mons[PM_GRID_BUG], u.desktop_entry_x, u.desktop_entry_y, 0);
        return '\033';
    }
    if (atoi(response) == 264) {
        d_level destination;
        struct trap *portal;
        destination.dnum = 0;
        destination.dlevel = 2;
        goto_level(&destination, FALSE, FALSE, FALSE);
        portal = maketrap(u.ux, u.uy, MAGIC_PORTAL);
        portal->dst.dnum = 0; portal->dst.dlevel = 1;
        destination.dlevel = 1;
        goto_level(&destination, FALSE, FALSE, FALSE);
        portal = maketrap(u.ux, u.uy, MAGIC_PORTAL);
        portal->dst.dnum = 0; portal->dst.dlevel = 2;
        destination.dlevel = 2;
        goto_level(&destination, FALSE, FALSE, TRUE);
        return '\033';
    }
    if (atoi(response) == 265) {
        teleds(xdnstair, ydnstair, FALSE);
        u.uhp = 1;
        HFumbling = 5L;
        return '>';
    }
    if (atoi(response) == 267) {
        d_level destination;
        destination.dnum = mines_dnum; destination.dlevel = 1;
        goto_level(&destination, TRUE, FALSE, FALSE);
        return '\033';
    }
    if (atoi(response) == 251) {
        d_level destination;
        destination.dnum = 0;
        destination.dlevel = 2;
        goto_level(&destination, FALSE, FALSE, FALSE);
        return '\033';
    }
#endif
    return atoi(response);
}

static int
bridge_poskey(int *x, int *y, int *mod)
{
    *x = *y = *mod = 0;
    return bridge_getch();
}

static char
bridge_yn(const char *query, const char *choices, CHAR_P def)
{
    /* Inventory choice parsing, ordering and object names come from nds_yn.c. */
    if (strstr(query, "What do you want") && strchr(query, '[')) {
        char *letters = _nds_parse_choices(query);
        if (letters && *letters) {
            winid window = create_nhwindow(NHW_MENU);
            anything value;
            menu_item *selection = 0;
            int i, count;
            char result = '\033';
            start_menu(window);
            for (i = 0; letters[i]; ++i) {
                const char *label;
                struct obj *object;
                char special[2];
                if (letters[i] == ' ' || letters[i] == '?') continue;
                value.a_void = 0; value.a_int = letters[i];
                object = obj_for_let(letters[i]);
                special[0] = letters[i]; special[1] = '\0';
                label = object ? doname(object) : letters[i] == '*'
                    ? "Something from your inventory" : letters[i] == '-'
                    || letters[i] == '.' ? "Nothing/your finger" : special;
                add_menu(window, NO_GLYPH, &value, letters[i], 0, 0, label, FALSE);
            }
            bridge_end_menu(window, query);
            count = select_menu(window, PICK_ONE, &selection);
            if (count > 0) result = (char) selection[0].item.a_int;
            if (selection) free(selection);
            destroy_nhwindow(window);
            return result;
        }
    }
    input_request(choices ? "question" : "direction", query, choices, def);
    {
        int answer = atoi(response);
        if (choices) {
            if (answer == '\033')
                answer = strchr(choices, 'q') ? 'q' : strchr(choices, 'n') ? 'n' : def;
            if (answer == ' ' || answer == '\n' || answer == '\r') answer = def;
        }
        return (char) answer;
    }
}

static void
bridge_getlin(const char *query, char *line)
{
    unsigned value;
    int index = 0;
    const char *p;
    input_request("text", query, "", 0);
    p = response;
    while (index < BUFSZ - 1 && sscanf(p, "%2x", &value) == 1) {
        line[index++] = (char) value;
        p += 2;
        if (!*p || *p == '\n') break;
    }
    line[index] = '\0';
}

static void
bridge_askname(void)
{
    char name[BUFSZ];
    bridge_getlin("Who are you?", name);
    (void) strncpy(plname, name, PL_NSIZ - 1);
    plname[PL_NSIZ - 1] = '\0';
}

static int
_nds_display_yes_no_prompt(char *query)
{
    char answer = bridge_yn(query, "yn", 'n');
    return answer == '\033' ? -1 : answer == 'y';
}

/* These two functions are extracted verbatim from the supplied DS port.
 * Input buffers are a DS hardware concern; the desktop has no queued macros.
 */
#define PROMPT_LAYER_WIDTH 80
#define DEBUG_PRINT(...) ((void) 0)
#define NULLFREE(pointer) do { if (pointer) free(pointer); pointer = 0; } while (0)
#define nds_input_buffer_is_empty() 1
#define nds_input_buffer_shiftall() ((char *) 0)
#include "nds_selection.inc"
#include "nds_extended.inc"

static winid
bridge_create(int type)
{
    int id = next_window++;
    if (id >= 128) panic("desktop window limit");
    window_type[id] = type;
    printf("{\"event\":\"create\",\"window\":%d,\"type\":%d}\n", id, type);
    return id;
}

static void
bridge_clear(winid id)
{
    printf("{\"event\":\"clear\",\"window\":%d}\n", id);
}

static void
bridge_destroy(winid id)
{
    printf("{\"event\":\"destroy\",\"window\":%d}\n", id);
    window_type[id] = 0;
    if (id == next_window - 1) --next_window;
}

static void
bridge_putstr(winid id, int attribute, const char *text)
{
    if (window_type[id] == NHW_MESSAGE) remember(text);
    printf("{\"event\":\"text\",\"window\":%d,\"attribute\":%d,\"text\":", id, attribute);
    quoted(text); printf("}\n");
}

static void
bridge_display(winid id, BOOLEAN_P blocking)
{
    printf("{\"event\":\"display\",\"window\":%d}\n", id);
    if (blocking) {
        bridge_mode();
        printf("{\"event\":\"input\",\"kind\":\"ack\",\"window\":%d,"
               "\"turn\":%ld}\n", id, moves);
        read_response();
    }
}

static void
bridge_raw(const char *text)
{
    printf("{\"event\":\"message\",\"text\":"); quoted(text); printf("}\n");
}

static void
bridge_file(const char *name, BOOLEAN_P complain)
{
    char line[BUFSZ];
    dlb *file = dlb_fopen(name, "r");
    winid id;
    if (!file) {
        if (complain) bridge_raw(name);
        return;
    }
    id = bridge_create(NHW_TEXT);
    while (dlb_fgets(line, sizeof line, file)) {
        line[strcspn(line, "\r\n")] = '\0';
        bridge_putstr(id, 0, line);
    }
    dlb_fclose(file);
    bridge_display(id, TRUE);
    bridge_destroy(id);
}

static void
bridge_start_menu(winid id)
{
    item_count[id] = 0;
    printf("{\"event\":\"menu_start\",\"window\":%d}\n", id);
}

static void
bridge_add_menu(winid id, int glyph, const ANY_P *identifier, CHAR_P key,
                CHAR_P group, int attribute, const char *text, BOOLEAN_P selected)
{
    int index = item_count[id]++;
    if (index >= 1024) panic("desktop menu limit");
    identifiers[id][index] = *identifier;
    printf("{\"event\":\"menu_item\",\"window\":%d,\"index\":%d,"
           "\"selectable\":%s,\"key\":%d,\"group\":%d,\"selected\":%s,\"text\":",
           id, index, identifier->a_void ? "true" : "false", (unsigned char) key,
           (unsigned char) group, selected ? "true" : "false");
    quoted(text); printf("}\n");
}

static void
bridge_end_menu(winid id, const char *prompt)
{
    printf("{\"event\":\"menu_end\",\"window\":%d,\"prompt\":", id);
    quoted(prompt); printf("}\n");
}

static int
bridge_select_menu(winid id, int how, MENU_ITEM_P **selection)
{
    int count, index, row;
    long quantity;
    char *position;
    printf("{\"event\":\"input\",\"kind\":\"menu\",\"window\":%d,"
           "\"how\":%d,\"turn\":%ld}\n", id, how, moves);
    read_response();
    count = (int) strtol(response, &position, 10);
    *selection = 0;
    if (count <= 0) return count < 0 ? -1 : 0;
    if (count > item_count[id]) panic("invalid desktop selection count");
    *selection = (menu_item *) alloc(count * sizeof(menu_item));
    for (index = 0; index < count; ++index) {
        row = (int) strtol(position, &position, 10);
        quantity = strtol(position, &position, 10);
        if (row < 0 || row >= item_count[id] || !identifiers[id][row].a_void)
            panic("invalid desktop menu row");
        (*selection)[index].item = identifiers[id][row];
        (*selection)[index].count = quantity;
    }
    return count;
}

static void
bridge_glyph(winid id, XCHAR_P x, XCHAR_P y, int glyph, int background)
{
    substitute_tiles(&u.uz);
    if (glyph >= 0 && glyph < MAX_GLYPH)
        printf("{\"event\":\"glyph\",\"x\":%d,\"y\":%d,\"glyph\":%d,"
               "\"tile\":%d,\"pet\":%s}\n", x, y, glyph, glyph2tile[glyph],
               glyph_is_pet(glyph) ? "true" : "false");
}

static void bridge_nop(void) { }
static void bridge_curs(winid id, int x, int y) {
    printf("{\"event\":\"cursor\",\"window\":%d,\"x\":%d,\"y\":%d}\n", id, x, y);
}
static void bridge_pad(int value) { }
static void bridge_clip(int x, int y) { }
static void bridge_suspend(const char *message) { }
static char *bridge_history(BOOLEAN_P first) {
    if (first) history_index = 0;
    return history_index < message_count ? messages[history_index++] : (char *) 0;
}
static void bridge_put_history(const char *text, BOOLEAN_P restore) { remember(text); }
static int bridge_previous(void) {
    int index;
    winid id = bridge_create(NHW_TEXT);
    for (index = 0; index < message_count; ++index) bridge_putstr(id, 0, messages[index]);
    bridge_display(id, TRUE);
    bridge_destroy(id);
    return 0;
}
static void
bridge_exit(const char *text)
{
    printf("{\"event\":\"exit\",\"text\":"); quoted(text); printf("}\n");
    fflush(stdout);
}

struct window_procs desktop_procs = {
    .name = "desktop", .wincap = WC_COLOR | WC_TILED_MAP,
    .wincap2 = 0,
    .win_init_nhwindows = bridge_init, .win_player_selection = nds_player_selection,
    .win_askname = bridge_askname, .win_get_nh_event = bridge_nop,
    .win_exit_nhwindows = bridge_exit, .win_suspend_nhwindows = bridge_suspend,
    .win_resume_nhwindows = bridge_nop, .win_create_nhwindow = bridge_create,
    .win_clear_nhwindow = bridge_clear, .win_display_nhwindow = bridge_display,
    .win_destroy_nhwindow = bridge_destroy, .win_curs = bridge_curs,
    .win_putstr = bridge_putstr, .win_putmixed = genl_putmixed,
    .win_display_file = bridge_file, .win_start_menu = bridge_start_menu,
    .win_add_menu = bridge_add_menu, .win_end_menu = bridge_end_menu,
    .win_select_menu = bridge_select_menu, .win_message_menu = genl_message_menu,
    .win_update_inventory = bridge_nop, .win_mark_synch = bridge_nop,
    .win_wait_synch = bridge_nop, .win_print_glyph = bridge_glyph,
#ifdef CLIPPING
    .win_cliparound = bridge_clip,
#endif
    .win_raw_print = bridge_raw, .win_raw_print_bold = bridge_raw,
    .win_nhgetch = bridge_getch, .win_nh_poskey = bridge_poskey,
    .win_nhbell = bridge_nop, .win_doprev_message = bridge_previous,
    .win_yn_function = bridge_yn, .win_getlin = bridge_getlin,
    .win_get_ext_cmd = nds_get_ext_cmd, .win_number_pad = bridge_pad,
    .win_delay_output = bridge_nop, .win_start_screen = bridge_nop,
    .win_end_screen = bridge_nop, .win_outrip = genl_outrip,
    .win_preference_update = genl_preference_update,
    .win_getmsghistory = bridge_history, .win_putmsghistory = bridge_put_history,
#ifdef STATUS_VIA_WINDOWPORT
    .win_status_init = genl_status_init, .win_status_finish = genl_status_finish,
    .win_status_enablefield = genl_status_enablefield,
    .win_status_update = genl_status_update,
#ifdef STATUS_HILITES
    .win_status_threshold = genl_status_threshold,
#endif
#endif
    .win_can_suspend = genl_can_suspend_no
};
