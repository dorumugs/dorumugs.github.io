---
layout: single
title:  "(7/10) 2D SRPG 만들기 선행학습 — Tiled 맵 에디터로 전장 만들기"
date: 2026-07-08 06:45:00 +0900
categories: coding
tag: [게임개발, SRPG, 파랜드택틱스, pygame, Tiled, pytmx, TMX, 맵에디터, 타일맵, Python]
author_profile: false
toc: true
header:
  image: /assets/images/2026-07-08-srpg-prereq/header-7.svg
  teaser: /assets/images/2026-07-08-srpg-prereq/header-7.svg
description: "2D SRPG 선행학습 7편. 배열 손편집을 졸업하고 무료 맵 에디터 Tiled 로 아이소메트릭 전장을 그립니다. 타일 커스텀 속성으로 지형 규칙을 데이터화하고, 오브젝트 레이어로 유닛 배치까지 — pytmx 로 읽어서 기존 코드에 그대로 꽂는 것까지 실행 코드로 정리했어요."
series: srpg-prereq
series_order: 7
---

{% include series-srpg-prereq.html current="7" %}

## Summary

6편 로드맵에서 "가장 급한 항목"으로 꼽았던 **맵 에디터**를 다룹니다. 지금까지 우리 맵은 파이썬 배열이었어요. 8×8 은 어떻게든 버티지만, 30×30 전장을 숫자로 손편집하는 건 고문에 가깝습니다. 맵 하나 만드는 데 하루가 걸리면 게임은 영원히 완성되지 않아요.

업계 표준 해법이 **[Tiled](https://www.mapeditor.org)** 라는 무료 오픈소스 맵 에디터입니다. 그림 그리듯 맵을 칠하고, 저장된 TMX 파일을 `pytmx` 라이브러리로 읽으면 — 2편의 배열, 4편의 이동 규칙, 6편의 유닛 배치가 전부 **에디터에서 만든 데이터**로 대체됩니다. 6편에서 배운 "코드는 규칙만, 콘텐츠는 데이터로" 원칙의 완결편이에요.

> 💡 이 글에서 다루는 것
> - Tiled 설치와 아이소메트릭(쿼터뷰) 맵 프로젝트 설정
> - 타일셋 준비 — 에셋 없이 코드로 만드는 마름모 타일셋
> - 타일 커스텀 속성 — cost·passable 을 에디터에서 정의하기
> - 오브젝트 레이어 — 유닛 스폰 위치를 맵에 심기
> - pytmx 로 읽기 — 기존 `enter_cost`·`spawn` 인터페이스에 그대로 연결
> - 아이소메트릭 오브젝트 좌표 함정 (그리드 = 픽셀 ÷ 타일높이)

<br>

<br>



## 1. 왜 맵 에디터인가

배열 손편집의 한계를 먼저 짚고 갑시다. 실전 SRPG 맵을 만들다 보면 배열로는 감당이 안 되는 요구가 줄줄이 나와요.

| 요구사항 | 배열 손편집 | Tiled |
|---|---|---|
| 30×30 이상 큰 맵 | 숫자 900개를 눈으로 관리 | 마우스로 드래그해서 칠하기 |
| 지형을 보면서 수정 | 머릿속으로 렌더링 | 실제 타일 그림으로 즉시 확인 |
| 겹층 구조 (바닥 + 장식 + 지붕) | 배열을 여러 개 관리 | 레이어 기능 내장 |
| 유닛·보물상자 배치 | 좌표 하드코딩 | 오브젝트 레이어로 클릭 배치 |
| 지형 속성 (비용·통행) | 코드의 딕셔너리 | 타일에 속성 직접 부착 |

Tiled 는 이 전부를 해결해주고, 결과물을 **TMX** 라는 XML 파일로 저장합니다. 게임 쪽에서는 이 파일을 읽기만 하면 돼요. 설치는 [공식 사이트](https://www.mapeditor.org)에서 받거나 맥이면 한 줄입니다.

```shell
brew install --cask tiled
```

읽기 라이브러리는 pygame 진영의 표준인 `pytmx` 를 씁니다.

```shell
pip install pytmx
```

<br>

<br>



## 2. 아이소메트릭 맵 프로젝트 만들기

Tiled 를 열고 **File → New → New Map** 에서 아래처럼 설정합니다. 2편에서 정한 우리 게임의 규격과 정확히 맞추는 게 포인트예요.

| 설정 | 값 | 이유 |
|---|---|---|
| Orientation | **Isometric** | 쿼터뷰 마름모 격자 |
| Tile size | **64 × 32** | 2편의 `TILE_W`, `TILE_H` 그대로 |
| Map size | 8 × 8 (연습용) | 실전에선 자유롭게 |
| Tile layer format | CSV | 사람이 읽기 쉬움 |

Orientation 을 Isometric 으로 두는 순간 에디터 화면 자체가 마름모 격자로 바뀝니다. 2편에서 손으로 짰던 좌표 변환을 에디터가 똑같이 쓰고 있는 거예요.

### 타일셋 — 에셋 없이 코드로

맵을 칠하려면 **타일셋**(타일 그림 모음 이미지)이 필요합니다. 3편에서 했던 것처럼, 에셋이 없어도 코드로 만들면 돼요. 풀밭·물·산 마름모 세 개를 가로로 붙인 이미지를 생성합니다.

```python
import pygame

pygame.init()
TILE_W, TILE_H = 64, 32
COLORS = [(106, 168, 79), (61, 133, 198), (127, 106, 79)]   # 풀밭/물/산

sheet = pygame.Surface((TILE_W * 3, TILE_H), pygame.SRCALPHA)
for i, color in enumerate(COLORS):
    x0 = i * TILE_W
    pts = [(x0 + TILE_W // 2, 0), (x0 + TILE_W - 1, TILE_H // 2),
           (x0 + TILE_W // 2, TILE_H - 1), (x0, TILE_H // 2)]
    pygame.draw.polygon(sheet, color, pts)
    pygame.draw.polygon(sheet, (30, 30, 46), pts, 1)
pygame.image.save(sheet, "tileset.png")
print(sheet.get_size())
```

```text
(192, 32)
```

이 `tileset.png` 를 Tiled 에서 **New Tileset** 으로 등록합니다(Tile size 64×32, "Embed in map" 체크). 이제 붓으로 칠하듯 전장을 그릴 수 있어요. 스탬프 브러시(B)로 한 칸씩, 버킷(F)으로 영역을 채우면서 2편의 그 맵을 에디터에서 다시 만들어봅시다. 배열 숫자를 칠 때와는 차원이 다른 속도에 놀랄 거예요.

<br>

<br>



## 3. 지형 규칙과 유닛 배치를 맵에 심기

여기서부터가 Tiled 의 진짜 힘입니다. 그림만 그리는 게 아니라 **게임 데이터를 맵 파일에 함께 심을** 수 있어요.

### 타일 커스텀 속성

타일셋 편집 화면에서 타일을 선택하고 **Custom Properties** 에 속성을 추가합니다. 4편의 이동 규칙을 그대로 옮겨요.

| 타일 | terrain (string) | cost (int) | passable (bool) |
|---|---|---|---|
| 풀밭 | grass | 1 | ✔ |
| 물 | water | 1 | ✘ |
| 산 | mountain | 2 | ✔ |

이제 "물은 못 지나감, 산은 비용 2" 라는 규칙이 코드의 `MOVE_COST` 딕셔너리가 아니라 **맵 데이터 안에** 삽니다. 새 지형(숲, 늪)을 추가할 때 코드 수정이 필요 없어져요.

### 오브젝트 레이어로 유닛 스폰

**Layer → New → Object Layer** 로 `units` 레이어를 만들고, 점(Insert Point) 도구로 유닛이 시작할 칸을 클릭해 찍습니다. 오브젝트마다 Name 에 유닛 템플릿 id(`knight`, `archer`, `goblin`)를, Custom Properties 에 `team` 속성(`ally`/`enemy`)을 넣어요. 6편의 `units.json` 템플릿과 이름으로 연결되는 구조입니다 — **어떤 유닛인지는 JSON 이, 어디에 서는지는 맵이** 결정해요.

저장하면 `battlefield.tmx` 파일이 생깁니다. 열어보면 사람이 읽을 수 있는 XML 이고, 타일 레이어는 우리가 쓰던 2차원 배열이 CSV 로 들어있어요.

> 💡 TMX 의 CSV 숫자는 우리 배열보다 전부 1씩 큽니다. GID(전역 타일 ID) 체계에서 **0 = 빈 칸**으로 예약돼 있고 첫 타일이 1부터 시작하기 때문이에요. pytmx 를 쓰면 이 변환을 알아서 해주니 직접 계산할 일은 없지만, 파일을 눈으로 볼 때 헷갈리지 않도록 알아두세요.

<br>

<br>



## 4. pytmx 로 읽기 — 기존 코드에 그대로 꽂기

이제 게임 쪽에서 읽습니다. pygame 용 로더인 `load_pygame` 을 쓰면 타일 이미지까지 Surface 로 바로 나와요.

```python
import pygame
from pytmx.util_pygame import load_pygame

pygame.init()
screen = pygame.display.set_mode((960, 540))    # ⚠️ load_pygame 보다 먼저!

tmx = load_pygame("battlefield.tmx")
print(tmx.orientation, tmx.width, "x", tmx.height, "타일,",
      tmx.tilewidth, "x", tmx.tileheight, "px")
```

```text
isometric 8 x 8 타일, 64 x 32 px
```

> ⚠️ `load_pygame` 은 내부에서 이미지를 화면 포맷으로 변환(3편의 `convert_alpha`)하기 때문에, 반드시 `pygame.display.set_mode()` **다음에** 호출해야 합니다. 순서를 바꾸면 "No video mode has been set" 에러가 나요.

타일 레이어는 `tiles()` 로 순회합니다. `(그리드 x, 그리드 y, 타일 이미지)` 가 나와서 2편의 렌더 루프에 바로 꽂혀요.

```python
ground = tmx.get_layer_by_name("ground")
tiles = list(ground.tiles())
print(len(tiles), "타일, 첫 타일:", tiles[0][:2], type(tiles[0][2]).__name__)
```

```text
64 타일, 첫 타일: (0, 0) Surface
```

3장에서 심은 커스텀 속성은 `get_tile_properties(gx, gy, 레이어번호)` 로 읽습니다. 물 타일이 있던 (1, 2) 를 조회해보면:

```python
props = tmx.get_tile_properties(1, 2, 0)
print({k: props[k] for k in ("terrain", "cost", "passable")})
```

```text
{'terrain': 'water', 'cost': 1, 'passable': False}
```

에디터에서 입력한 속성이 타입까지 살아서 그대로 나와요. 이걸로 4편의 `enter_cost` 를 다시 쓰면 — 시그니처가 같으니 `movement_range` 와 `astar` 는 **한 글자도 안 바꾸고** 그대로 돕니다.

```python
def enter_cost(gx, gy):
    p = tmx.get_tile_properties(gx, gy, 0)
    return p["cost"] if p and p["passable"] else None

print("풀밭(0,0):", enter_cost(0, 0), "/ 물(1,2):", enter_cost(1, 2),
      "/ 산(4,0):", enter_cost(4, 0))
```

```text
풀밭(0,0): 1 / 물(1,2): None / 산(4,0): 2
```

<br>

<br>



## 5. 오브젝트 → 유닛 스폰, 그리고 좌표 함정

오브젝트 레이어는 `tmx.objects` 로 순회하는데, 여기에 이 글 최대의 함정이 숨어 있습니다. **아이소메트릭 맵의 오브젝트 좌표는 화면 픽셀이 아니라, "그리드 좌표 × 타일높이"** 예요. 그리드 칸으로 되돌리려면 x, y **둘 다** 타일높이(32)로 나눕니다. 타일 너비(64)가 아니라요.

```python
for obj in tmx.objects:
    gx, gy = int(obj.x // tmx.tileheight), int(obj.y // tmx.tileheight)
    print(obj.name, obj.properties["team"], "→ 그리드", (gx, gy))
```

```text
knight ally → 그리드 (1, 1)
archer ally → 그리드 (2, 6)
goblin enemy → 그리드 (6, 2)
```

에디터에서 찍은 세 유닛이 정확한 그리드 칸으로 돌아왔어요. `obj.name` 이 6편 템플릿의 id 와 이어지므로, 스폰은 이 한 줄이면 끝입니다.

```python
units = [spawn(templates, obj.name, int(obj.x // tmx.tileheight),
               int(obj.y // tmx.tileheight), obj.properties["team"])
         for obj in tmx.objects]
```

> 🚨 이 좌표 규약을 모르고 `obj.x` 를 그대로 화면 좌표로 쓰면 유닛이 맵 밖 엉뚱한 곳에 스폰됩니다. "Tiled 아이소메트릭 오브젝트 좌표는 투영 전 좌표계" — 아이소메트릭 게임 개발자들이 한 번씩 밟고 지나가는 유명한 함정이니 여기서 미리 접종하고 가세요.

<br>

<br>



## 6. 실습 — 에디터로 만든 전장을 게임에 띄우기

전부 합칩니다. 2편의 데모와 화면은 같지만, 이번엔 **맵도 지형 규칙도 유닛 배치도 전부 TMX 에서** 옵니다. 코드에는 배열도, `MOVE_COST` 도, 스폰 좌표도 없어요.

```python
import pygame
from pytmx.util_pygame import load_pygame

pygame.init()
screen = pygame.display.set_mode((960, 540))
pygame.display.set_caption("SRPG 선행학습 7편 — Tiled 전장")
clock = pygame.time.Clock()

tmx = load_pygame("battlefield.tmx")
TILE_W, TILE_H = tmx.tilewidth, tmx.tileheight
ROWS, COLS = tmx.height, tmx.width
ORIGIN_X, ORIGIN_Y = 480, 120

def grid_to_screen(gx, gy):
    return ((gx - gy) * (TILE_W // 2) + ORIGIN_X,
            (gx + gy) * (TILE_H // 2) + ORIGIN_Y)

def tile_center(gx, gy):
    sx, sy = grid_to_screen(gx, gy)
    return sx, sy + TILE_H // 2

def enter_cost(gx, gy):                      # 4편 인터페이스, TMX 버전
    p = tmx.get_tile_properties(gx, gy, 0)
    return p["cost"] if p and p["passable"] else None

ground = tmx.get_layer_by_name("ground")
spawns = [(o.name, int(o.x // TILE_H), int(o.y // TILE_H),
           o.properties["team"]) for o in tmx.objects]

running = True
while running:
    dt = clock.tick(60) / 1000
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((30, 30, 46))
    for gx, gy, image in ground.tiles():     # 마름모 꼭대기 기준 보정
        sx, sy = grid_to_screen(gx, gy)
        screen.blit(image, (sx - TILE_W // 2, sy))
    for name, gx, gy, team in spawns:
        cx, cy = tile_center(gx, gy)
        color = (94, 198, 214) if team == "ally" else (198, 94, 150)
        pygame.draw.circle(screen, color, (cx, cy - 8), 10)
    pygame.display.flip()

pygame.quit()
```

실행하면 Tiled 에서 그린 전장이 그대로 뜨고, 에디터에서 찍은 위치에 유닛들이 서 있습니다. 이제부터 맵 수정은 게임을 끄지 않아도 돼요 — **Tiled 에서 고치고 저장하고, 게임을 재시작하면 끝**. 4·5편의 이동 범위·전투 코드는 `enter_cost` 와 스폰이 위처럼 바뀐 것 말고는 그대로 얹으면 됩니다.

> 🎉 이로써 6편의 "코드는 규칙만, 콘텐츠는 데이터로" 원칙이 완성됐습니다. 유닛과 스킬은 JSON 이, 전장과 배치는 TMX 가, 코드는 규칙만 — 이게 상용 게임과 같은 구조예요. 챕터 10개짜리 게임을 만들려면 TMX 파일 10개를 그리면 됩니다.

<br>

<br>



## 7. 정리

- [x] 맵은 에디터로 — Tiled 아이소메트릭 프로젝트, 타일 크기는 게임 규격(64×32)과 일치
- [x] 타일셋도 코드로 생성 가능 — 에셋은 나중에 갈아끼우면 된다
- [x] 지형 규칙(cost·passable)은 타일 커스텀 속성으로 — 코드에서 딕셔너리 제거
- [x] 유닛 배치는 오브젝트 레이어로 — 템플릿 id 는 name, 진영은 team 속성
- [x] 아이소메트릭 오브젝트 좌표 함정 — 그리드 = 좌표 ÷ 타일높이 (둘 다!)
- [x] `enter_cost` 인터페이스를 지킨 덕에 4·5편 알고리즘은 무수정 재사용

게임 루프에서 시작해 에디터 파이프라인까지, 파랜드 택틱스 같은 게임의 뼈대를 이루는 부품이 전부 손에 들어왔어요. 그런데 지금 게임엔 정보가 안 보입니다 — HP 는 막대뿐이고, 명중률은 콘솔에만 찍히고, 이동 후에 공격할지 대기할지 고를 수도 없죠. 다음 편에서 이걸 해결합니다.

일단 오늘은 여기까지.....
다음 글에서는 로드맵의 2번, UI 와 메뉴를 정리해볼게요.

---

**← 이전 글:** [(6/10) 2D SRPG 만들기 선행학습 — 데이터 설계와 세이브/로드](/coding/SRPG_만들기_선행학습_데이터_세이브/) ｜ **다음 글 →:** [(8/10) 2D SRPG 만들기 선행학습 — UI 와 메뉴, 정보를 보여주는 기술](/coding/SRPG_만들기_선행학습_UI_메뉴/)
