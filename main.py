import pygame
import sys
import random
import heapq
import math
import time

ROWS = 10
COLS = 10
CELL = 52
MARGIN = 2
TITLE_H = 40
BOTTOM_H = 120
PANEL_W = 220
GRID_TOP = TITLE_H
GRID_LEFT = 40
GRID_W = COLS * (CELL + MARGIN) + MARGIN
GRID_H = ROWS * (CELL + MARGIN) + MARGIN
WIN_W = GRID_LEFT + GRID_W + PANEL_W + 20
WIN_H = GRID_TOP + GRID_H + BOTTOM_H + 10

C_WIN_BG       = (230, 230, 230)
C_GRID_BG      = (200, 200, 200)
C_CELL_EMPTY   = (225, 225, 225)
C_WALL         = (60,  60,  60)
C_START        = (40,  170, 60)
C_TARGET       = (220, 60,  60)
C_FRONTIER     = (255, 220, 0)
C_EXPLORED     = (50,  100, 180)
C_PATH         = (0,   200, 80)
C_PATH_LINE    = (0,   160, 60)
C_PATH_ARROW   = (255, 255, 255)
C_PATH_GLOW    = (0,   255, 120)
C_TEXT_DARK    = (20,  20,  20)
C_TEXT_LIGHT   = (255, 255, 255)
C_TITLE_TEXT   = (30,  30,  30)
C_BTN_BG       = (245, 245, 245)
C_BTN_BORDER   = (160, 160, 160)
C_BTN_ACTIVE   = (180, 200, 240)
C_BTN_HOVER    = (210, 220, 240)
C_PANEL_BG     = (245, 245, 250)
C_PANEL_BORDER = (180, 180, 200)
C_ITEM_TOP     = (80,  180, 90)
C_ITEM_BORDER  = (160, 140, 0)
C_PANEL_HEADER = (60,  80,  160)
C_FWD_ITEM     = (255, 220, 0)

DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ALGO_NAMES = ["GBFS", "A*", "Weighted A*", "A* Tree"]
HEUR_NAMES = ["Manhattan", "Euclidean"]


def heuristic_manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def heuristic_euclidean(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def get_neighbors(pos, walls, rows, cols):
    r, c = pos
    nb = []
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in walls:
            nb.append((nr, nc))
    return nb


def reconstruct_path(came_from, start, target):
    path = []
    node = target
    while node != start:
        path.append(node)
        node = came_from[node]
    path.append(start)
    path.reverse()
    return path


def path_blocked(path, walls):
    return any(p in walls for p in path)


def make_walls(rows, cols, start, target, density=0.20):
    walls = set()
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in (start, target) and random.random() < density:
                walls.add((r, c))
    return walls


def gbfs_step(state, start, target, walls, rows, cols, heuristic):
    if state is None:
        h = heuristic(start, target)
        return {
            "pq":        [(h, 0, start)],
            "visited":   {start},
            "came_from": {start: None},
            "order":     {start: 0},
            "counter":   [1],
            "tie":       [1],
            "frontier":  {start},
            "explored":  set(),
            "path":      None,
            "done":      False,
            "g":         {start: 0},
            "h":         {start: h},
            "f":         {start: h},
        }
    s = state
    if s["done"] or not s["pq"]:
        s["done"] = True
        if s["path"] is None:
            s["path"] = []
        return s
    _, _, node = heapq.heappop(s["pq"])
    s["frontier"].discard(node)
    if node in s["explored"]:
        return s
    s["explored"].add(node)
    if node == target:
        s["done"] = True
        s["path"] = reconstruct_path(s["came_from"], start, target)
        return s
    for nb in get_neighbors(node, walls, rows, cols):
        if nb not in s["visited"]:
            s["visited"].add(nb)
            s["came_from"][nb] = node
            h_val = heuristic(nb, target)
            s["h"][nb] = h_val
            s["g"][nb] = s["g"][node] + 1
            s["f"][nb] = h_val
            s["tie"][0] += 1
            heapq.heappush(s["pq"], (h_val, s["tie"][0], nb))
            s["frontier"].add(nb)
            s["order"][nb] = s["counter"][0]
            s["counter"][0] += 1
    return s


def astar_step(state, start, target, walls, rows, cols, heuristic, weight=1.0):
    if state is None:
        h = heuristic(start, target)
        return {
            "pq":        [(weight * h, 0, start)],
            "g":         {start: 0},
            "h":         {start: h},
            "f":         {start: weight * h},
            "came_from": {start: None},
            "visited":   {start},
            "closed":    set(),
            "order":     {start: 0},
            "counter":   [1],
            "tie":       [1],
            "frontier":  {start},
            "explored":  set(),
            "path":      None,
            "done":      False,
            "weight":    weight,
        }
    s = state
    if s["done"] or not s["pq"]:
        s["done"] = True
        if s["path"] is None:
            s["path"] = []
        return s
    _, _, node = heapq.heappop(s["pq"])
    s["frontier"].discard(node)
    if node in s["closed"]:
        return s
    s["closed"].add(node)
    s["explored"].add(node)
    if node == target:
        s["done"] = True
        s["path"] = reconstruct_path(s["came_from"], start, target)
        return s
    for nb in get_neighbors(node, walls, rows, cols):
        if nb in s["closed"]:
            continue
        new_g = s["g"][node] + 1
        if nb not in s["g"] or new_g < s["g"][nb]:
            s["g"][nb] = new_g
            h_val = heuristic(nb, target)
            s["h"][nb] = h_val
            f_val = new_g + s["weight"] * h_val
            s["f"][nb] = f_val
            s["came_from"][nb] = node
            s["visited"].add(nb)
            s["tie"][0] += 1
            heapq.heappush(s["pq"], (f_val, s["tie"][0], nb))
            s["frontier"].add(nb)
            if nb not in s["order"]:
                s["order"][nb] = s["counter"][0]
                s["counter"][0] += 1
    return s


def weighted_astar_step(state, start, target, walls, rows, cols, heuristic):
    return astar_step(state, start, target, walls, rows, cols, heuristic, weight=2.0)


def astar_tree_step(state, start, target, walls, rows, cols, heuristic):
    if state is None:
        h = heuristic(start, target)
        return {
            "pq":        [(h, 0, start)],
            "g":         {start: 0},
            "h":         {start: h},
            "f":         {start: h},
            "came_from": {start: None},
            "visited":   {start},
            "order":     {start: 0},
            "counter":   [1],
            "tie":       [1],
            "frontier":  {start},
            "explored":  set(),
            "path":      None,
            "done":      False,
        }
    s = state
    if s["done"] or not s["pq"]:
        s["done"] = True
        if s["path"] is None:
            s["path"] = []
        return s
    _, _, node = heapq.heappop(s["pq"])
    s["frontier"].discard(node)
    s["explored"].add(node)
    if node == target:
        s["done"] = True
        s["path"] = reconstruct_path(s["came_from"], start, target)
        return s
    for nb in get_neighbors(node, walls, rows, cols):
        new_g = s["g"].get(node, 0) + 1
        if nb not in s["g"] or new_g < s["g"][nb]:
            s["g"][nb] = new_g
            h_val = heuristic(nb, target)
            s["h"][nb] = h_val
            f_val = new_g + h_val
            s["f"][nb] = f_val
            s["came_from"][nb] = node
            s["visited"].add(nb)
            s["tie"][0] += 1
            heapq.heappush(s["pq"], (f_val, s["tie"][0], nb))
            s["frontier"].add(nb)
            if nb not in s["order"]:
                s["order"][nb] = s["counter"][0]
                s["counter"][0] += 1
    return s


def get_ds_items(algo_state, sel_algo):
    if algo_state is None:
        return None, []
    s = algo_state
    label_map = {
        0: "OPEN LIST  (h-value)",
        1: "OPEN LIST  (f = g+h)",
        2: "OPEN LIST  (f = g+W·h)",
        3: "OPEN LIST  (f = g+h) [Tree]",
    }
    label = label_map.get(sel_algo, "OPEN LIST")
    raw = sorted(s.get("pq", []), key=lambda x: x[0])
    items = []
    for entry in raw:
        node = entry[2]
        g_v  = s.get("g", {}).get(node, 0)
        h_v  = s.get("h", {}).get(node, 0)
        extra = f"h={h_v:.1f}" if sel_algo == 0 else f"g={g_v} h={h_v:.1f}"
        items.append({
            "text":  f"({node[0]},{node[1]})",
            "color": C_FWD_ITEM,
            "extra": extra,
        })
    return label, items


def draw_ds_panel(surface, font_hdr, font_item, font_small,
                  sel_algo, algo_state, panel_x, panel_y, panel_w, panel_h):
    pygame.draw.rect(surface, C_PANEL_BG,
                     (panel_x, panel_y, panel_w, panel_h), border_radius=6)
    pygame.draw.rect(surface, C_PANEL_BORDER,
                     (panel_x, panel_y, panel_w, panel_h), 2, border_radius=6)
    ds_label, items = get_ds_items(algo_state, sel_algo)
    if ds_label is None:
        defaults = ["OPEN LIST (h)", "OPEN LIST (f)",
                    "OPEN LIST (f,W=2)", "OPEN LIST (f) Tree"]
        ds_label = defaults[sel_algo]
    hdr_surf = font_hdr.render(ds_label, True, C_PANEL_HEADER)
    surface.blit(hdr_surf,
                 (panel_x + panel_w // 2 - hdr_surf.get_width() // 2, panel_y + 8))
    size_txt = font_small.render(f"size: {len(items)}", True, (100, 100, 130))
    surface.blit(size_txt,
                 (panel_x + panel_w - size_txt.get_width() - 8, panel_y + 10))
    pygame.draw.line(surface, C_PANEL_BORDER,
                     (panel_x + 8, panel_y + 28),
                     (panel_x + panel_w - 8, panel_y + 28), 1)
    item_h = 26
    item_pad = 3
    visible_start = panel_y + 34
    max_visible = (panel_h - 50) // (item_h + item_pad)
    display_items = items[:max_visible]
    for idx, item in enumerate(display_items):
        iy = visible_start + idx * (item_h + item_pad)
        is_top = (idx == 0)
        bg_col = C_ITEM_TOP if is_top else item["color"]
        pygame.draw.rect(surface, bg_col,
                         (panel_x + 8, iy, panel_w - 16, item_h), border_radius=4)
        border_col = (50, 130, 50) if is_top else C_ITEM_BORDER
        pygame.draw.rect(surface, border_col,
                         (panel_x + 8, iy, panel_w - 16, item_h), 1, border_radius=4)
        if is_top:
            tag = font_small.render("NEXT", True, C_TEXT_LIGHT)
            surface.blit(tag, (panel_x + 12, iy + item_h // 2 - tag.get_height() // 2))
            txt = font_item.render(item["text"], True, C_TEXT_LIGHT)
        else:
            txt = font_item.render(item["text"], True, C_TEXT_DARK)
        surface.blit(txt, (panel_x + 46, iy + item_h // 2 - txt.get_height() // 2))
        if item["extra"]:
            ext_col = (220, 240, 220) if is_top else (80, 80, 120)
            ext = font_small.render(item["extra"], True, ext_col)
            surface.blit(ext,
                         (panel_x + panel_w - ext.get_width() - 12,
                          iy + item_h // 2 - ext.get_height() // 2))
    if len(items) > max_visible:
        more = font_small.render(f"+ {len(items) - max_visible} more...", True, (120, 120, 150))
        surface.blit(more,
                     (panel_x + panel_w // 2 - more.get_width() // 2,
                      panel_y + panel_h - 18))


def draw_cell_value(surface, font, r, c,
                    order, frontier, explored, path_set,
                    walls, dyn_obs, start, target,
                    grid_left, grid_top, cell, margin,
                    algo_state, sel_algo):
    x = grid_left + margin + c * (cell + margin)
    y = grid_top  + margin + r * (cell + margin)
    pos = (r, c)
    if pos == start:
        bg, label, text_col = C_START, "s", C_TEXT_LIGHT
    elif pos == target:
        bg, label, text_col = C_TARGET, "t", C_TEXT_LIGHT
    elif pos in walls or pos in dyn_obs:
        bg, label, text_col = C_WALL, "-1", C_TEXT_LIGHT
    elif pos in path_set and pos not in (start, target):
        bg, label, text_col = C_PATH, "", C_TEXT_LIGHT
    elif pos in explored:
        bg = C_EXPLORED
        text_col = C_TEXT_LIGHT
        if algo_state and sel_algo == 0:
            h_v = algo_state.get("h", {}).get(pos, "")
            label = f"h{h_v:.0f}" if isinstance(h_v, float) else str(h_v)
        elif algo_state and sel_algo in (1, 2, 3):
            f_v = algo_state.get("f", {}).get(pos, "")
            label = f"f{f_v:.0f}" if isinstance(f_v, float) else (f"f{f_v}" if f_v != "" else str(order.get(pos, "")))
        else:
            label = str(order.get(pos, ""))
    elif pos in frontier:
        bg = C_FRONTIER
        text_col = C_TEXT_DARK
        if algo_state and sel_algo == 0:
            h_v = algo_state.get("h", {}).get(pos, "")
            label = f"h{h_v:.0f}" if isinstance(h_v, float) else str(h_v)
        elif algo_state and sel_algo in (1, 2, 3):
            f_v = algo_state.get("f", {}).get(pos, "")
            label = f"f{f_v:.0f}" if isinstance(f_v, float) else (f"f{f_v}" if f_v != "" else "")
        else:
            label = ""
    else:
        bg, label, text_col = C_CELL_EMPTY, "0", C_TEXT_DARK
    pygame.draw.rect(surface, bg, (x, y, cell, cell))
    pygame.draw.rect(surface, C_GRID_BG, (x, y, cell, cell), 1)
    txt = font.render(label, True, text_col)
    surface.blit(txt, (x + cell // 2 - txt.get_width() // 2,
                       y + cell // 2 - txt.get_height() // 2))


def cell_center(r, c, grid_left, grid_top, cell, margin):
    x = grid_left + margin + c * (cell + margin) + cell // 2
    y = grid_top  + margin + r * (cell + margin) + cell // 2
    return x, y


def draw_arrow(surface, color, cx, cy, dr, dc, size):
    angle  = math.atan2(dr, dc)
    tip_x  = cx + int(math.cos(angle) * size * 0.52)
    tip_y  = cy + int(math.sin(angle) * size * 0.52)
    left_x = cx + int(math.cos(angle + 2.4) * size * 0.28)
    left_y = cy + int(math.sin(angle + 2.4) * size * 0.28)
    righ_x = cx + int(math.cos(angle - 2.4) * size * 0.28)
    righ_y = cy + int(math.sin(angle - 2.4) * size * 0.28)
    pygame.draw.polygon(surface, color, [(tip_x, tip_y), (left_x, left_y), (righ_x, righ_y)])


def draw_path_overlay(surface, path_list, start, target,
                      grid_left, grid_top, cell, margin, anim_t):
    if not path_list or len(path_list) < 2:
        return
    centers = [cell_center(r, c, grid_left, grid_top, cell, margin) for r, c in path_list]
    pulse = abs((anim_t % 1.0) - 0.5) * 2.0
    line_alpha = int(180 + 60 * pulse)
    line_surf = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
    for i in range(len(centers) - 1):
        pygame.draw.line(line_surf, (*C_PATH_LINE, line_alpha), centers[i], centers[i + 1], 5)
    surface.blit(line_surf, (0, 0))
    for i, (r, c) in enumerate(path_list):
        cx, cy = centers[i]
        if (r, c) in (start, target):
            continue
        glow_r = int(cell * 0.44 + cell * 0.08 * pulse)
        glow_surf = pygame.Surface((glow_r * 2 + 2, glow_r * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*C_PATH_GLOW, int(60 + 40 * pulse)),
                           (glow_r + 1, glow_r + 1), glow_r)
        surface.blit(glow_surf, (cx - glow_r - 1, cy - glow_r - 1))
        if i < len(path_list) - 1:
            nr, nc = path_list[i + 1]
            draw_arrow(surface, C_PATH_ARROW, cx, cy, nr - r, nc - c, cell)
        step_font = pygame.font.SysFont("arial", 10, bold=True)
        step_surf = step_font.render(str(i), True, (20, 20, 20))
        surface.blit(step_surf, (cx - step_surf.get_width() // 2,
                                 cy - step_surf.get_height() // 2))


class Button:
    def __init__(self, rect, label, font):
        self.rect    = pygame.Rect(rect)
        self.label   = label
        self.font    = font
        self.active  = False
        self.hovered = False

    def draw(self, surf):
        bg = (C_BTN_ACTIVE if self.active
              else C_BTN_HOVER if self.hovered
              else C_BTN_BG)
        pygame.draw.rect(surf, bg, self.rect, border_radius=4)
        pygame.draw.rect(surf, C_BTN_BORDER, self.rect, 1, border_radius=4)
        t = self.font.render(self.label, True, C_TEXT_DARK)
        surf.blit(t, (self.rect.centerx - t.get_width()  // 2,
                      self.rect.centery - t.get_height() // 2))

    def hit(self, pos):          return self.rect.collidepoint(pos)
    def update_hover(self, pos): self.hovered = self.rect.collidepoint(pos)


def main():
    pygame.init()
    try:
        font_cell        = pygame.font.SysFont("arial", 13, bold=True)
        font_title       = pygame.font.SysFont("arial", 18, bold=False)
        font_btn         = pygame.font.SysFont("arial", 13, bold=True)
        font_status      = pygame.font.SysFont("arial", 13)
        font_panel_hdr   = pygame.font.SysFont("arial", 11, bold=True)
        font_panel_item  = pygame.font.SysFont("arial", 12, bold=True)
        font_panel_small = pygame.font.SysFont("arial", 10)
    except Exception:
        font_cell        = pygame.font.SysFont(None, 18)
        font_title       = pygame.font.SysFont(None, 22)
        font_btn         = pygame.font.SysFont(None, 17)
        font_status      = pygame.font.SysFont(None, 16)
        font_panel_hdr   = pygame.font.SysFont(None, 15)
        font_panel_item  = pygame.font.SysFont(None, 16)
        font_panel_small = pygame.font.SysFont(None, 13)

    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("AI Pathfinder – Informed Search Visualizer")
    clock = pygame.time.Clock()

    START  = (7, 1)
    TARGET = (6, 7)

    static_walls = make_walls(ROWS, COLS, START, TARGET)
    dyn_obs      = set()
    all_walls    = lambda: static_walls | dyn_obs

    sel_algo     = 0
    sel_heur     = 0
    algo_state   = None
    searching    = False
    step_delay   = 0.06
    last_step    = 0.0
    last_obs     = time.time()
    obs_interval = 3.0
    status       = "Select algorithm and press START"
    anim_t       = 0.0

    bottom_y = GRID_TOP + GRID_H + 8

    algo_btns = []
    bw, bh = 82, 28
    total  = len(ALGO_NAMES) * (bw + 6) - 6
    ax0    = GRID_LEFT + (GRID_W - total) // 2
    for i, name in enumerate(ALGO_NAMES):
        b = Button((ax0 + i * (bw + 6), bottom_y, bw, bh), name, font_btn)
        if i == sel_algo:
            b.active = True
        algo_btns.append(b)

    heur_y    = bottom_y + bh + 5
    heur_btns = []
    hw, hh = 110, 24
    hx0    = GRID_LEFT
    for i, name in enumerate(HEUR_NAMES):
        b = Button((hx0 + i * (hw + 6), heur_y, hw, hh), name, font_btn)
        if i == sel_heur:
            b.active = True
        heur_btns.append(b)

    ctrl_y    = heur_y + hh + 5
    btn_start = Button((GRID_LEFT,       ctrl_y, 85,  28), "START",     font_btn)
    btn_reset = Button((GRID_LEFT + 91,  ctrl_y, 85,  28), "RESET",     font_btn)
    btn_walls = Button((GRID_LEFT + 182, ctrl_y, 110, 28), "NEW WALLS", font_btn)

    spd_labels = [("SLOW", 0.12), ("MED", 0.05), ("FAST", 0.01), ("MAX", 0.0)]
    spd_idx  = 1
    spd_btns = []
    sx0 = GRID_LEFT + 310
    for si, (sl, _) in enumerate(spd_labels):
        sb = Button((sx0 + si * 66, ctrl_y, 60, 28), sl, font_btn)
        if si == spd_idx:
            sb.active = True
        spd_btns.append(sb)

    step_delay = spd_labels[spd_idx][1]

    panel_x = GRID_LEFT + GRID_W + 10
    panel_y = GRID_TOP
    panel_h = GRID_H

    def get_heuristic():
        return heuristic_manhattan if sel_heur == 0 else heuristic_euclidean

    def reset():
        nonlocal algo_state, searching, status
        algo_state = None
        searching  = False
        status     = "Press START to begin"

    def start():
        nonlocal algo_state, searching, status
        algo_state = None
        searching  = True
        status     = f"Running {ALGO_NAMES[sel_algo]} ({HEUR_NAMES[sel_heur]})..."

    def do_step():
        nonlocal algo_state, searching, status
        w = all_walls()
        h = get_heuristic()
        if sel_algo == 0:
            algo_state = gbfs_step(algo_state, START, TARGET, w, ROWS, COLS, h)
        elif sel_algo == 1:
            algo_state = astar_step(algo_state, START, TARGET, w, ROWS, COLS, h)
        elif sel_algo == 2:
            algo_state = weighted_astar_step(algo_state, START, TARGET, w, ROWS, COLS, h)
        elif sel_algo == 3:
            algo_state = astar_tree_step(algo_state, START, TARGET, w, ROWS, COLS, h)
        if algo_state and algo_state["done"]:
            searching = False
            p = algo_state.get("path") or []
            if len(p) > 1:
                cost = algo_state.get("g", {}).get(TARGET, len(p) - 1)
                status = (f"Path found: {len(p)} nodes | "
                          f"cost={cost:.1f} | "
                          f"explored={len(algo_state.get('explored', set()))}")
            else:
                status = "No path found!"

    while True:
        now   = time.time()
        dt_ms = clock.tick(60)
        anim_t += dt_ms / 1000.0
        mp = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, b in enumerate(algo_btns):
                    if b.hit(mp):
                        sel_algo = i
                        for bb in algo_btns:
                            bb.active = False
                        algo_btns[i].active = True
                        reset()

                for i, b in enumerate(heur_btns):
                    if b.hit(mp):
                        sel_heur = i
                        for bb in heur_btns:
                            bb.active = False
                        heur_btns[i].active = True
                        reset()

                if btn_start.hit(mp):
                    start()
                if btn_reset.hit(mp):
                    reset()
                    dyn_obs.clear()
                if btn_walls.hit(mp):
                    static_walls.clear()
                    static_walls.update(make_walls(ROWS, COLS, START, TARGET))
                    dyn_obs.clear()
                    reset()

                for si, sb in enumerate(spd_btns):
                    if sb.hit(mp):
                        spd_idx    = si
                        step_delay = spd_labels[si][1]
                        for ssb in spd_btns:
                            ssb.active = False
                        spd_btns[si].active = True

                gx = mp[0] - GRID_LEFT - MARGIN
                gy = mp[1] - GRID_TOP  - MARGIN
                if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                    gc  = gx // (CELL + MARGIN)
                    gr  = gy // (CELL + MARGIN)
                    pos = (gr, gc)
                    if 0 <= gr < ROWS and 0 <= gc < COLS and pos not in (START, TARGET):
                        if pos in static_walls:
                            static_walls.discard(pos)
                        else:
                            static_walls.add(pos)
                        reset()

        for b in algo_btns:  b.update_hover(mp)
        for b in heur_btns:  b.update_hover(mp)
        btn_start.update_hover(mp)
        btn_reset.update_hover(mp)
        btn_walls.update_hover(mp)
        for sb in spd_btns:  sb.update_hover(mp)

        if searching:
            if now - last_obs > obs_interval:
                for _ in range(40):
                    r2 = random.randint(0, ROWS - 1)
                    c2 = random.randint(0, COLS - 1)
                    p2 = (r2, c2)
                    if (p2 not in (START, TARGET)
                            and p2 not in static_walls
                            and p2 not in dyn_obs):
                        dyn_obs.add(p2)
                        break
                last_obs = now
                if algo_state and algo_state.get("path"):
                    if path_blocked(algo_state["path"], all_walls()):
                        start()
            if step_delay == 0.0 or (now - last_step >= step_delay):
                do_step()
                last_step = now

        screen.fill(C_WIN_BG)

        title_str  = f"{ALGO_NAMES[sel_algo]}  [{HEUR_NAMES[sel_heur]}]"
        title_surf = font_title.render(title_str, True, C_TITLE_TEXT)
        screen.blit(title_surf,
                    (WIN_W // 2 - title_surf.get_width() // 2,
                     (TITLE_H - title_surf.get_height()) // 2))

        pygame.draw.rect(screen, C_GRID_BG, (GRID_LEFT, GRID_TOP, GRID_W, GRID_H))

        order_map    = algo_state["order"]                 if algo_state and "order" in algo_state else {}
        frontier_set = algo_state.get("frontier", set())  if algo_state else set()
        explored_set = algo_state.get("explored", set())  if algo_state else set()
        path_list    = (algo_state.get("path") or [])     if algo_state else []
        path_set     = set(path_list)

        for r in range(ROWS):
            for c in range(COLS):
                draw_cell_value(
                    screen, font_cell, r, c,
                    order_map, frontier_set, explored_set, path_set,
                    static_walls, dyn_obs,
                    START, TARGET,
                    GRID_LEFT, GRID_TOP, CELL, MARGIN,
                    algo_state, sel_algo,
                )

        draw_path_overlay(screen, path_list, START, TARGET,
                          GRID_LEFT, GRID_TOP, CELL, MARGIN, anim_t)

        draw_ds_panel(screen, font_panel_hdr, font_panel_item, font_panel_small,
                      sel_algo, algo_state, panel_x, panel_y, PANEL_W, panel_h)

        leg_y  = panel_y + panel_h - 68
        legend = [
            (C_START,    "S=start"),
            (C_TARGET,   "T=goal"),
            (C_FRONTIER, "Frontier"),
            (C_EXPLORED, "Visited"),
            (C_PATH,     "Path"),
            (C_WALL,     "Wall"),
        ]
        for li, (col, lbl) in enumerate(legend):
            lx = panel_x + 8 + (li % 2) * (PANEL_W // 2)
            ly = leg_y   + (li // 2) * 20
            pygame.draw.rect(screen, col, (lx, ly, 12, 12), border_radius=2)
            lt = font_panel_small.render(lbl, True, C_TEXT_DARK)
            screen.blit(lt, (lx + 16, ly))

        for b in algo_btns:  b.draw(screen)
        for b in heur_btns:  b.draw(screen)
        btn_start.draw(screen)
        btn_reset.draw(screen)
        btn_walls.draw(screen)
        for sb in spd_btns:  sb.draw(screen)

        pygame.draw.rect(screen, C_WIN_BG, (0, WIN_H - 22, WIN_W, 22))
        status_surf = font_status.render(status, True, (60, 60, 60))
        screen.blit(status_surf,
                    (WIN_W // 2 - status_surf.get_width() // 2, WIN_H - 20))

        pygame.display.flip()


if __name__ == "__main__":
    main()