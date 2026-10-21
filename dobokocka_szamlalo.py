import math
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk



DEBUG = True                  # True: a konzolra írja, miért esett ki egy jelölt

# Kockalap (a kontúrból készített maszk)
FACE_ERODE_REL = 0.10          # a lapmaszk zsugorítása a kocka oldalához képest

# Pöttyszűrés
MIN_PIP_AREA_ABS = 12          # abszolút minimum (pixel), ennél kisebb zaj
MIN_PIP_AREA_REL = 0.004       # a kocka lapterületéhez viszonyított minimum
MAX_PIP_AREA_REL = 0.12        # a kocka lapterületéhez viszonyított maximum
MIN_CIRCULARITY = 0.60         # 4*pi*A/P^2 (1.0 = tökéletes kör)
MIN_CIRCLE_FILL = 0.60         # A / (pi * r_befoglaló^2)
EDGE_MARGIN_REL = 0.02         # min. távolság a lapmaszk szélétől (kocka oldal arányában)
MERGE_DIST_FACTOR = 1.5        # ennél közelebbi pöttyök = duplikátum (sugár szorzó)
MAX_PIP_AREA_RATIO = 2.5       # legnagyobb / legkisebb pötty területaránya

# Pöttykeresés árnyékban (black-hat: a lokális fényességtől független)
PIP_KERNEL_REL = 0.35          # a black-hat kernel mérete a kocka oldalához képest
PIP_SEARCH_SIDE = 160          # ennél nagyobb kockánál a pöttykeresés kicsinyített képen fut
MIN_PIP_CONTRAST = 10          # min. fényességkülönbség a pötty és a lap között
PIP_REL_CONTRAST = 0.30        # ... és a lap fényességének legalább ennyi hányada

# Kockajelölt-szűrés (mintás háttér ellen)
MIN_SOLIDITY = 0.85            # kontúr terület / konvex burok terület
MIN_RECTANGULARITY = 0.75      # kontúr terület / minAreaRect terület (forgatásra érzéketlen)
MIN_SIDE_RATIO = 0.60          # minAreaRect rövidebb / hosszabb oldala
MAX_FACE_STD = 12.0            # a lap finom textúrájának (robusztus) szórása; a lassú árnyék nem számít
MAX_FACE_EDGE_DENSITY = 0.12   # Canny-élek aránya a kocka lapján (pöttyök nélkül)
MIN_PIP_DIAMETER_REL = 0.06    # pötty átmérő / kocka oldal
MAX_PIP_DIAMETER_REL = 0.35

# Pöttyalapú keresés (összeérő vagy árnyékban lévő kockák "mentése")
PIP_TO_SIDE = 5.3              # kocka oldala ~ PIP_TO_SIDE * pöttyátmérő
GROUP_SPAN_REL = 0.72          # egy kocka pöttyei legfeljebb ekkora távolságra vannak (oldal arányában)
SCALE_TOLERANCE = 0.20         # a becsült és a feltételezett kockaméret megengedett eltérése
MIN_BOUNDARY_CONTRAST = 15     # a lap és a környezete fényességkülönbsége (szürkeárnyalat)
MIN_BOUNDARY_SECTORS = 4       # hány a 12 irányból mutasson ilyen különbséget
RESCUE_MAX_SIDE = 1000         # a pöttykeresést ekkorára kicsinyített képen végezzük (sebesség)
MAX_RESCUE_PIPS = 300          # ennél több pötty-jelölt: textúra, nem kockák


def _odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1


def _fill_outside(gray_crop, inside):
    """A kockán kívüli részt a lap mediánjával töltjük ki, hogy ne adjon élt."""
    filled = gray_crop.copy()
    filled[~inside] = int(np.median(gray_crop[inside]))
    return filled


def _pip_candidates(filled, face_mask, kernel, min_area, max_area, margin):
    """Köralakú, sötét foltok keresése a lapon (árnyékra érzéketlen black-hat módszerrel).

    A black-hat = (zárás - kép): csak a kernelnél kisebb sötét részleteket emeli ki,
    a nagy, lassan változó árnyékot nem. Visszatér: [(cx, cy, sugár, terület), ...]
    """
    blur = cv2.GaussianBlur(filled, (3, 3), 0)
    k = _odd(max(9, kernel))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, se)
    blackhat = cv2.subtract(closed, blur).astype(np.float32)
    threshold = np.maximum(MIN_PIP_CONTRAST, closed.astype(np.float32) * PIP_REL_CONTRAST)
    dark = ((blackhat > threshold) * 255).astype(np.uint8)
    dark = cv2.bitwise_and(dark, face_mask)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # RETR_LIST: a másik folt lyukában ülő pöttyöket (pl. résháló között álló kocka) is látni kell.
    contours, _ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    dist = cv2.distanceTransform(face_mask, cv2.DIST_L2, 3) if margin is not None else None
    candidates = []
    for c in contours:
        # 1) Túl kicsi (vagy túl nagy) kontúrok kiszűrése
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue

        # 2) Köralakúság ellenőrzése
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if radius <= 0 or area / (math.pi * radius * radius) < MIN_CIRCLE_FILL:
            continue

        # 3) A kocka széléhez közeli találatok kizárása
        if dist is not None:
            pts = c.reshape(-1, 2)
            if dist[pts[:, 1], pts[:, 0]].min() < margin:
                continue

        candidates.append((cx, cy, radius, area))
    return candidates


def _merge_duplicates(candidates):
    """4) Az egymáshoz közeli, duplikált találatok összevonása."""
    candidates = sorted(candidates, key=lambda p: p[3], reverse=True)   # nagyobbak előre
    merged = []
    for cx, cy, r, area in candidates:
        duplicate = False
        for mx, my, mr, _ in merged:
            if math.hypot(cx - mx, cy - my) < max(r, mr) * MERGE_DIST_FACTOR:
                duplicate = True
                break
        if not duplicate:
            merged.append((cx, cy, r, area))
    return merged


def _pip_mask(shape, pips):
    mask = np.zeros(shape[:2], np.uint8)
    for cx, cy, r, _ in pips:
        cv2.circle(mask, (int(cx), int(cy)), int(math.ceil(r)) + 2, 255, -1)
    return mask


def _detect_pips(gray_crop, face_mask, side):
    """Pöttyök keresése egyetlen kocka lapján.

    Visszatér: (pöttylista, pötty-maszk, kitöltött szürke kép).
    """
    inside = face_mask > 0
    face_area = int(np.count_nonzero(inside))
    if face_area == 0:
        return [], np.zeros_like(face_mask), gray_crop
    filled = _fill_outside(gray_crop, inside)

    # Nagy kockánál a pöttykeresést kicsinyített képen végezzük (a nagy kernelű
    # black-hat különben lassú), majd a találatokat visszaskálázzuk.
    f = min(1.0, PIP_SEARCH_SIDE / float(side))
    if f < 1.0:
        small = cv2.resize(filled, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(face_mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        candidates = _pip_candidates(
            small, small_mask, side * f * PIP_KERNEL_REL,
            max(MIN_PIP_AREA_ABS * f * f, face_area * f * f * MIN_PIP_AREA_REL),
            face_area * f * f * MAX_PIP_AREA_REL,
            max(2.0, side * f * EDGE_MARGIN_REL),
        )
        candidates = [(cx / f, cy / f, r / f, area / (f * f)) for cx, cy, r, area in candidates]
    else:
        candidates = _pip_candidates(
            filled, face_mask, side * PIP_KERNEL_REL,
            max(MIN_PIP_AREA_ABS, face_area * MIN_PIP_AREA_REL),
            face_area * MAX_PIP_AREA_REL,
            max(2.0, side * EDGE_MARGIN_REL),
        )
    merged = _merge_duplicates(candidates)
    return merged, _pip_mask(gray_crop.shape, merged), filled


def _contour_looks_like_die(contour, area):
    """Alakvizsgálat (forgatásra érzéketlen). Visszatér: (ok, oldalhossz, kiesés oka)."""
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area == 0 or area / hull_area < MIN_SOLIDITY:
        return False, 0.0, "solidity"
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    if rw == 0 or rh == 0:
        return False, 0.0, "minAreaRect=0"
    if area / (rw * rh) < MIN_RECTANGULARITY:
        return False, 0.0, "rectangularity"
    if min(rw, rh) / max(rw, rh) < MIN_SIDE_RATIO:
        return False, 0.0, "side ratio"
    return True, math.sqrt(rw * rh), ""


def _face_is_clean(filled, region_mask, pip_mask, side):
    """A kocka lapja egyenletes-e (a pöttyök kivételével)?

    Mintás háttér (rács, csempe, szövet) sok finom élt és textúrát ad.
    A lassú fényesség-változás (árnyék) nem számít: a szórást a
    simított képpel csökkentett (felüláteresztett) képen, robusztus módon
    (medián abszolút eltérés) mérjük, így egy árnyékél sem rontja el.
    Visszatér: (ok, szórás, éldenzitás).
    """
    face = (region_mask > 0) & (pip_mask == 0)
    n = int(face.sum())
    if n < 20:
        return False, 0.0, 0.0
    sigma = max(1.5, side * 0.03)
    img = filled.astype(np.float32)
    # Normalizált simítás: a pöttyök és a lapon kívüli rész nem torzítja a helyi átlagot.
    valid = face.astype(np.float32)
    local = cv2.GaussianBlur(img * valid, (0, 0), sigma) / np.maximum(
        cv2.GaussianBlur(valid, (0, 0), sigma), 1e-3)
    highpass = img - local
    std = float(1.4826 * np.median(np.abs(highpass[face])))
    edges = cv2.Canny(cv2.GaussianBlur(filled, (3, 3), 0), 40, 110)
    density = float(np.count_nonzero(edges[face])) / n
    return (std <= MAX_FACE_STD and density <= MAX_FACE_EDGE_DENSITY), std, density


def _crop_with_mask(gray, contour, pad=2):
    """A kontúr körüli kivágás és a kitöltött kontúrmaszk."""
    x, y, w, h = cv2.boundingRect(contour)
    ox, oy = max(0, x - pad), max(0, y - pad)
    crop = gray[oy:y + h + pad, ox:x + w + pad]
    mask = np.zeros(crop.shape, np.uint8)
    cv2.drawContours(mask, [contour - np.array([[ox, oy]])], -1, 255, -1)
    return crop, mask, ox, oy


def _analyze_die(gray, contour):
    """Egy kontúr ellenőrzése egyetlen kockaként.

    Visszatér: (kocka | None, kiesés oka, info). Az info a kocka tényleges
    oldalhosszát és a pöttyök átlagos sugarát tartalmazza (a későbbi fázisnak).
    """
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    if min(w, h) < 15 or h == 0:
        return None, "túl kicsi", None
    if not 0.60 <= w / float(h) <= 1.55:
        return None, "oldalarány", None

    # 5) Téves kockakontúrok (mintás háttér) kiszűrése alak alapján
    ok, side, why = _contour_looks_like_die(contour, area)
    if not ok:
        return None, why, None

    # Lapmaszk: a kitöltött kontúr, kissé zsugorítva (a szélek helyett a belső rész).
    crop, mask, _, _ = _crop_with_mask(gray, contour)
    k = max(3, int(side * FACE_ERODE_REL))
    face_mask = cv2.erode(mask, np.ones((k, k), np.uint8))

    pips, pip_mask, filled = _detect_pips(crop, face_mask, side)
    if not 1 <= len(pips) <= 6:
        return None, f"pöttyszám: {len(pips)}", None

    # Pöttyméret-konzisztencia: egy kockán a pöttyök egyforma nagyok.
    areas = [p[3] for p in pips]
    if max(areas) / min(areas) > MAX_PIP_AREA_RATIO:
        return None, "pöttyméretek eltérnek", None

    # Pöttyátmérő a kocka méretéhez képest.
    mean_diameter = 2.0 * float(np.mean([p[2] for p in pips]))
    if not MIN_PIP_DIAMETER_REL <= mean_diameter / side <= MAX_PIP_DIAMETER_REL:
        return None, "pötty/kocka méretarány", None

    # 5) A kocka lapja egyenletes-e (nem mintás háttér)?
    clean, std, density = _face_is_clean(filled, face_mask, pip_mask, side)
    if not clean:
        return None, f"lap nem egyenletes (szórás {std:.1f}, él {density:.3f})", None

    info = {"side": side, "radius": float(np.mean([p[2] for p in pips]))}
    return (x, y, w, h, len(pips)), "", info


def _group_pips(pips, side):
    """Pöttyök csoportosítása kockákká (egymást követő, legközelebbi párok összevonásával).

    Két csoport akkor vonható össze, ha az összes pötty páronkénti távolsága
    belefér egy kocka pöttymezőjébe (GROUP_SPAN_REL * oldal). Ez a forgatásra
    érzéketlen, és az átlós (2-es, 3-as) mintákat is egyben tartja.
    """
    limit = GROUP_SPAN_REL * side
    groups = [[p] for p in pips]

    def span_ok(a, b):
        for p in a:
            for q in b:
                if math.hypot(p[0] - q[0], p[1] - q[1]) > limit:
                    return False
        return True

    def gap(a, b):
        return min(math.hypot(p[0] - q[0], p[1] - q[1]) for p in a for q in b)

    while True:
        best, best_d = None, None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                d = gap(groups[i], groups[j])
                if d <= limit and (best_d is None or d < best_d) and span_ok(groups[i], groups[j]):
                    best, best_d = (i, j), d
        if best is None:
            return groups
        i, j = best
        groups[i] = groups[i] + groups[j]
        del groups[j]


def _has_boundary(gray, cx, cy, side):
    """Van-e a pöttycsoport körül kockaszerű határ?

    Egy valódi kocka lapja a környezetétől (asztal, árnyék, rés a szomszéd
    kocka mellett) fényességben eltér. Egy sima felületen lévő foltcsoportnál
    nincs ilyen határ. A kocka belsejét (0,40-0,47 oldal) 12 szögtartományban
    összevetjük a 0,55-1,0 oldal közötti gyűrűk mediánjával (a medián miatt a
    pöttyök és a vékony árnyékélek nem zavarnak). Visszatér: a találó irányok száma.
    """
    r_out = int(side * 1.0) + 1
    x1, y1 = max(0, int(cx) - r_out), max(0, int(cy) - r_out)
    x2, y2 = min(gray.shape[1], int(cx) + r_out + 1), min(gray.shape[0], int(cy) + r_out + 1)
    patch = gray[y1:y2, x1:x2].astype(np.float32)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    dist = np.hypot(xx - cx, yy - cy) / side
    sector = ((np.arctan2(yy - cy, xx - cx) + math.pi) / (2 * math.pi) * 12).astype(int) % 12
    inner = (dist >= 0.40) & (dist <= 0.47)
    bins = [(lo, lo + 0.05) for lo in np.arange(0.55, 1.0, 0.05)]
    hits = 0
    for k in range(12):
        a = patch[inner & (sector == k)]
        if a.size < 5:
            continue
        inner_med = float(np.median(a))
        for lo, hi in bins:
            b = patch[(dist >= lo) & (dist < hi) & (sector == k)]
            if b.size >= 5 and abs(float(np.median(b)) - inner_med) >= MIN_BOUNDARY_CONTRAST:
                hits += 1
                break
    return hits


def _rescue_dice(gray, dice, hint, debug=False):
    """A maszkokkal el nem fogadott kockák keresése a pöttyök alapján.

    Árnyékban vagy összeérő kockáknál a kockák körvonala nem megbízható, de a
    pöttyök (black-hat módszerrel) még jól kereshetők. Az eddig elfogadott
    kockákon kívül eső pöttyöket csoportosítjuk, és minden csoportból becsült
    kockát készítünk. Ha nincs elfogadott kocka (nincs méretbecslés), több
    kockaméretet is kipróbálunk.
    """
    h, w = gray.shape
    scale = min(1.0, RESCUE_MAX_SIDE / float(min(h, w)))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
    full = np.full(small.shape, 255, np.uint8)

    if hint:
        sides = [hint["side"] * scale]
    else:
        sides = []
        s = 40.0
        while s <= 0.4 * min(small.shape):
            sides.append(s)
            s *= 1.3

    found = []
    for S in sides:
        r_exp = hint["radius"] * scale if hint else S / (2.0 * PIP_TO_SIDE)
        area_exp = math.pi * r_exp * r_exp
        cands = _pip_candidates(small, full, S * PIP_KERNEL_REL,
                                0.4 * area_exp, 2.5 * area_exp, None)
        if not cands:
            continue
        r_med = float(np.median([c[2] for c in cands]))
        cands = [c for c in cands if 0.7 * r_med <= c[2] <= 1.4 * r_med]
        if not hint:
            # Csak akkor fogadjuk el a méretet, ha a talált pöttyök is ezt támasztják alá.
            if abs(PIP_TO_SIDE * 2.0 * r_med - S) > SCALE_TOLERANCE * S:
                continue
            S = PIP_TO_SIDE * 2.0 * r_med
        pips = _merge_duplicates(cands)

        if not pips or len(pips) > MAX_RESCUE_PIPS:
            continue

        def inside_die(px, py, d, factor):
            return math.hypot(px - (d[0] + d[2] / 2.0), py - (d[1] + d[3] / 2.0)) < factor * min(d[2], d[3])

        for group in _group_pips(pips, S):
            if len(group) > 6:
                continue
            xs = [p[0] for p in group]
            ys = [p[1] for p in group]
            cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
            fx, fy = cx / scale, cy / scale

            # a) A csoport egy már elfogadott kocka: a pöttyszámot ellenőrizzük.
            #    (Ha a kocka körvonalát részben elvitte az árnyék, a széli pöttyök kiestek.)
            match = next((i for i, d in enumerate(dice) if inside_die(fx, fy, d, 0.2)), None)
            if match is not None:
                d = dice[match]
                if len(group) > d[4]:
                    if debug:
                        print(f"[pöttyszám javítva] {d[:4]}: {d[4]} -> {len(group)}")
                    dice[match] = (d[0], d[1], d[2], d[3], len(group))
                continue

            # b) Részben elfogadott kockához tartozó pöttyök: nem új kocka.
            if any(inside_die(p[0] / scale, p[1] / scale, d, 0.45) for p in group for d in dice):
                continue

            # c) Új kocka: egyforma pöttyök és egyenletes lap kell.
            areas = [p[3] for p in group]
            if max(areas) / min(areas) > MAX_PIP_AREA_RATIO:
                continue

            # A lap vizsgálata a középpont körüli körön (forgatott kockánál sem lóg ki).
            region = np.zeros(small.shape, np.uint8)
            cv2.circle(region, (int(cx), int(cy)), max(3, int(S * 0.48)), 255, -1)
            clean, std, density = _face_is_clean(small, region, _pip_mask(small.shape, group), S)
            if not clean:
                if debug:
                    print(f"[mentés] csoport eldobva ({len(group)} pötty): "
                          f"nem egyenletes (szórás {std:.1f}, él {density:.3f})")
                continue

            hits = _has_boundary(small, cx, cy, S)
            if hits < MIN_BOUNDARY_SECTORS:
                if debug:
                    print(f"[mentés] csoport eldobva ({len(group)} pötty): nincs kockahatár ({hits}/12 irány)")
                continue

            x1, y1 = max(0, int(round(cx - S / 2))), max(0, int(round(cy - S / 2)))
            x2, y2 = min(small.shape[1], int(round(cx + S / 2))), min(small.shape[0], int(round(cy + S / 2)))
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            new_die = (int(x1 / scale), int(y1 / scale),
                       int((x2 - x1) / scale), int((y2 - y1) / scale), len(group))
            if not any(_is_same_die(new_die, d) for d in dice + found):
                found.append(new_die)
    return found


def _build_masks(image, gray):
    """Több, egymást kiegészítő kockamaszk.

    Az első a színes és világos (fehér) kockákat adja pontosan, a második
    lokális fényesség alapján dolgozik, ezért árnyékban is működik.
    """
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]

    # 1) Színes kockák és világos, kevéssé telített (pl. fehér) kockák.
    colored_mask = cv2.inRange(saturation, 45, 255)
    bright_mask = cv2.inRange(blurred, 175, 255)
    low_sat_mask = cv2.inRange(saturation, 0, 75)
    white_mask = cv2.bitwise_and(bright_mask, low_sat_mask)
    base = cv2.bitwise_or(colored_mask, white_mask)

    # 2) Árnyékálló: a kocka lapja a közvetlen környezeténél világosabb.
    #    A lokális átlagot kicsinyített képen számoljuk (nagy képen így is gyors),
    #    az összehasonlítás viszont teljes felbontáson történik.
    f = min(1.0, 600.0 / min(h, w))
    small = cv2.resize(blurred, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1.0 else blurred
    sigma = max(2.0, 0.0375 * min(small.shape))
    local_mean = cv2.GaussianBlur(small.astype(np.float32), (0, 0), sigma)
    if f < 1.0:
        local_mean = cv2.resize(local_mean, (w, h), interpolation=cv2.INTER_LINEAR)
    local_bright = ((blurred.astype(np.float32) > local_mean + 10) * 255).astype(np.uint8)
    # Alacsony telítettség: a színes kockák az első maszkban vannak, itt a szürkés/fehér
    # kockák árnyékos változatait keressük.
    shadow_white = cv2.bitwise_and(local_bright, cv2.inRange(saturation, 0, 75))

    # 3) A kettő uniója (olyan kockákhoz, amelyek részben árnyékosak).
    union = cv2.bitwise_or(base, shadow_white)
    return [base, shadow_white, union]


def _mask_contours(mask, image_shape):
    # A nyitás kerneljét a képmérethez igazítjuk: a vékony háttérminták
    # (vonalak, rács) eltűnnek, a kockák megmaradnak.
    open_k = max(3, int(min(image_shape[:2]) * 0.015) // 2 * 2 + 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _is_same_die(a, b):
    """Két találat ugyanaz a kocka-e (a középpontok közelsége alapján)?"""
    ax, ay = a[0] + a[2] / 2.0, a[1] + a[3] / 2.0
    bx, by = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
    side = min(a[2], a[3], b[2], b[3])
    return math.hypot(ax - bx, ay - by) < 0.4 * side


def _add_if_new(dice, die):
    if not any(_is_same_die(die, other) for other in dice):
        dice.append(die)
        return True
    return False


def find_dice_and_pips(image, debug=None):
    """Több dobókocka felismerése egyszerre, árnyékos képen is."""
    debug = DEBUG if debug is None else debug

    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_area = image.shape[0] * image.shape[1]

    dice = []
    infos = []
    # 1. fázis: minden maszk minden foltját önálló kockaként vizsgáljuk.
    for mask in _build_masks(image, gray):
        for contour in _mask_contours(mask, image.shape):
            area = cv2.contourArea(contour)
            if area < image_area * 0.0015 or area > image_area * 0.18:
                continue
            die, reason, info = _analyze_die(gray, contour)
            if die is not None:
                if _add_if_new(dice, die):
                    infos.append(info)
            elif debug:
                print(f"[kiesett] {cv2.boundingRect(contour)}: {reason}")

    # 2. fázis: árnyékban lévő vagy összeérő kockák keresése a pöttyök alapján.
    hint = None
    if infos:
        hint = {"side": float(np.median([i["side"] for i in infos])),
                "radius": float(np.median([i["radius"] for i in infos]))}
    for die in _rescue_dice(gray, dice, hint, debug):
        if _add_if_new(dice, die) and debug:
            print(f"[megmentve a pöttyök alapján] {die}")

    dice.sort(key=lambda d: (d[1], d[0]))
    for index, (x, y, w, h, value) in enumerate(dice, start=1):
        # Jól látható zöld keret minden felismert kocka körül.
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 10)
        # Az értéket a keret fölött, kontrasztos címkén jelenítjük meg.
        label = str(value)
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 3, 2
        )
        text_y = max(text_h + 4, y - 8)

        cv2.putText(
            output, label, (x + 3, text_y), cv2.FONT_HERSHEY_SIMPLEX,
            3, (0, 0, 0), 2, cv2.LINE_AA
        )
    return output, dice


class DiceCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dobókocka-számláló")
        self.root.geometry("900x700")
        self.original_image = None
        self.photo = None

        header = tk.Label(
            root, text="Dobókocka-számláló alkalmazás",
            font=("Arial", 18, "bold")
        )
        header.pack(pady=12)

        controls = tk.Frame(root)
        controls.pack(pady=5)

        tk.Button(
            controls, text="Kép betöltése", command=self.load_image,
            font=("Arial", 11), padx=12, pady=6
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            controls, text="Felismerés újra", command=self.process_image,
            font=("Arial", 11), padx=12, pady=6
        ).pack(side=tk.LEFT, padx=5)

        self.image_label = tk.Label(
            root, text="Tölts be egy képet a kezdéshez.",
            bg="#eeeeee", width=90, height=25
        )
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        self.result_label = tk.Label(
            root, text="Dobókockák: –    Összeg: –",
            font=("Arial", 14, "bold")
        )
        self.result_label.pack(pady=10)


    def load_image(self):
        path = filedialog.askopenfilename(
            title="Kép kiválasztása",
            filetypes=[
                ("Képfájlok", "*.png *.jpg *.jpeg *.bmp"),
                ("Minden fájl", "*.*")
            ]
        )
        if not path:
            return

        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Hiba", "A kiválasztott kép nem olvasható.")
            return

        self.original_image = image
        self.process_image()

    def process_image(self):
        if self.original_image is None:
            messagebox.showinfo("Információ", "Először tölts be egy képet.")
            return

        result, dice = find_dice_and_pips(self.original_image)
        count = len(dice)
        total = sum(d[4] for d in dice)

        self.result_label.config(
            text=f"Dobókockák: {count}    Összeg: {total}"
        )

        if count == 0:
            messagebox.showwarning(
                "Nincs felismerhető kocka",
                "A program nem talált egyértelmű dobókockát. "
                "Próbálj világosabb, kontrasztosabb képet."
            )

        # Tkinterben RGB képet jelenítünk meg.
        rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        pil_image.thumbnail((850, 470))
        self.photo = ImageTk.PhotoImage(pil_image)
        self.image_label.config(image=self.photo, text="")


if __name__ == "__main__":
    root = tk.Tk()
    app = DiceCounterApp(root)
    root.mainloop()