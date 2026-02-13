# Snake Game Training Plan

## Hedef
Natural language ile "yılan oyunu yap" dediğimizde çalışan WASM kodu üretmek.

## Mevcut Durum
- ✅ Basit matematik fonksiyonları çalışıyor (add, multiply, max)
- ❌ Game-specific fonksiyonlar yok

## Snake Oyunu Gereksinimleri

### Phase 1: Temel Fonksiyonlar (Bu Notebook)

| Kategori | Fonksiyonlar | Adet |
|----------|--------------|------|
| **Math** | abs, min, max, clamp, mod, sign, wrap | 7 |
| **Point** | pack_point, unpack_x/y, point_equal, manhattan_distance, move_point | 6 |
| **Direction** | get_dx/dy, opposite, turn_left/right, is_opposite | 6 |
| **Collision** | in_bounds, hit_wall, wrap_x/y | 4 |
| **Random** | lcg_next, random_range, random_position | 3 |
| **State** | make_game_state, get_score/length, is_alive, add_score, set_dead | 7 |
| **Grid** | grid_index, index_to_x/y, grid_size | 4 |
| **Snake** | move_snake_head, check_food_collision, calculate_score | 4 |
| **Array** | circular_index, next/prev_index, buffer_full/empty/length | 6 |
| **Toplam** | | **47 fonksiyon** |

Her fonksiyon için 5-6 instruction varyasyonu → **~280 training örneği**

### Phase 2: Kompozit Fonksiyonlar (Sonraki Adım)

```
game_tick(state, direction) → new_state
check_self_collision(snake_body, head) → bool
spawn_food(seed, snake_body) → food_position
render_to_buffer(game_state, buffer) → void
```

### Phase 3: Host Integration

WASM Memory Layout:
```
0x0000 - 0x00FF: Game State (score, length, direction, alive)
0x0100 - 0x01FF: Snake Body (packed points, max 64 segments)
0x0200 - 0x02FF: Grid State (20x20 = 400 cells)
0x0300 - 0x03FF: Render Buffer
```

Host-side (Python/JS):
- Input handling (keyboard → direction)
- Timing (game loop @ 10fps)
- Rendering (buffer → terminal/canvas)

## Colab Kullanımı

### Step 1: Data Generation (Free CPU)
1. Notebook'u aç
2. Step 1-6'yı çalıştır
3. Google Drive'a kaydet

### Step 2: Training (H100 GPU)
1. Runtime → Change runtime type → H100
2. Step 7-8'i çalıştır
3. Model'i Drive'a kaydet

### Step 3: Deploy (Gradio)
1. Step 9'u çalıştır
2. Public URL al
3. `zerolang_cli.py` ile test et

## Beklenen Sonuç

```bash
$ python zerolang_cli.py --api https://xxx.gradio.live

zerolang> gen Implement: int pack_point(int x, int y)
[✓] Generated WAT

zerolang> run pack_point(5, 10)
Result: 327690  # (5 << 16) | 10

zerolang> gen Implement: int move_snake_head(int head, int dir, int w, int h)
[✓] Generated WAT

zerolang> run move_snake_head(327690, 1, 20, 20)
Result: 393226  # x=6, y=10
```

## Dosyalar

- `notebooks/Snake_Game_Training.ipynb` - Ana training notebook
- `src/docs/snake-game-plan.md` - Bu döküman

## Zaman Çizelgesi

1. Data generation: ~10 dk (CPU)
2. Training: ~30-45 dk (H100)
3. Test & Deploy: ~5 dk
