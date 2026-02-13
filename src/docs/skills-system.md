# ZeroLang Skills System

Modüler yetenek sistemi - her skill pack bağımsız olarak eklenebilir/çıkarılabilir.

## Mevcut Skill Packs

| Skill | Fonksiyon | Açıklama |
|-------|-----------|----------|
| `core_math` | 24 | Temel matematik: add, multiply, factorial, gcd... |
| `geometry` | 17 | 2D geometri: point, vector, distance, collision |
| `game_utils` | 19 | Oyun: direction, bounds, grid, movement |
| `random` | 8 | PRNG: lcg, xorshift, range, hash |
| `data_structures` | 14 | Array: circular buffer, stack, index ops |
| `bitwise` | 16 | Bit manipulation: flags, packing, popcount |
| `state_management` | 10 | Game state: score, lives, level |
| `snake_game` | 10 | Snake-specific: move, eat, grow |
| **Toplam** | **118** | |

Her fonksiyon için ~6 instruction varyasyonu → **~700 training örneği**

## Skill Pack Yapısı

```python
SKILL_EXAMPLE = {
    "name": "skill_name",
    "description": "What this skill does",
    "functions": [
        {
            "instruction": "Implement: int func(int x) - description",
            "code": "int func(int x) { return x * 2; }",
            "variations": ["double x", "multiply by 2"]
        },
        # ...
    ]
}
```

## Yeni Skill Ekleme

1. Notebook'ta yeni `SKILL_XXX` dict oluştur
2. `ALL_SKILLS`'e ekle
3. `SELECTED_SKILLS`'e dahil et
4. Data üret ve train et

## Örnek: Physics Skill Pack (Gelecek)

```python
SKILL_PHYSICS = {
    "name": "physics",
    "description": "Basic physics calculations",
    "functions": [
        {
            "instruction": "Implement: int velocity(int dist, int time)",
            "code": "int velocity(int dist, int time) { return time > 0 ? dist / time : 0; }",
            "variations": ["speed calculation", "distance over time"]
        },
        {
            "instruction": "Implement: int gravity_step(int vy, int g)",
            "code": "int gravity_step(int vy, int g) { return vy + g; }",
            "variations": ["apply gravity", "accelerate downward"]
        },
        # ...
    ]
}
```

## Örnek: String Skill Pack (Gelecek)

```python
SKILL_STRING = {
    "name": "string",
    "description": "String/character operations",
    "functions": [
        {
            "instruction": "Implement: int is_digit(int c)",
            "code": "int is_digit(int c) { return (c >= '0' && c <= '9') ? 1 : 0; }",
            "variations": ["check if character is digit"]
        },
        {
            "instruction": "Implement: int to_upper(int c)",
            "code": "int to_upper(int c) { return (c >= 'a' && c <= 'z') ? c - 32 : c; }",
            "variations": ["convert to uppercase"]
        },
        # ...
    ]
}
```

## Hedef Oyunlar

### Snake (Mevcut)
- ✅ core_math
- ✅ geometry
- ✅ game_utils
- ✅ random
- ✅ data_structures
- ✅ state_management
- ✅ snake_game

### Pong (Gelecek)
- core_math
- geometry
- game_utils
- physics (velocity, collision response)
- state_management

### Tetris (Gelecek)
- core_math
- geometry
- game_utils
- data_structures (2D array rotation)
- state_management
- tetris_game (piece rotation, line clear)

### Space Invaders (Gelecek)
- core_math
- geometry
- game_utils
- physics
- entity_system (spawn, destroy, update)
- state_management

## Kullanım

### Colab'da
```python
# Sadece ihtiyacın olan skill'leri seç
SELECTED_SKILLS = [
    "core_math",
    "geometry", 
    "game_utils",
]
```

### CLI'da Test
```bash
python zerolang_cli.py --api https://xxx.gradio.live

zerolang> gen Implement: int pack_point(int x, int y)
zerolang> run pack_point(10, 20)
Result: 655380
```
