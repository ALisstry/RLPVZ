import os
import random

import torch

from simenv import SimPVZEnv
from simenv.model import DDQNNetwork, transform_observation
from simenv.pvz_sim import config


pygame = None

MODEL_PATH = r"saved\ddqn\20260629_202423\sim_ddqn.pt"
POLICY = "model"  # "model", "random", or "wait"
FPS = 15
MAX_ACTIONS = 1000
DEVICE = "auto"

CELL_SIZE = 75
GRID_OFFSET_X = 100
GRID_OFFSET_Y = 90
INFO_HEIGHT = 145
SPRITE_SCALE = {
    "plant": (70, 70),
    "zombie": (66, 96),
    "zombie_flag": (82, 82),
    "projectile": (24, 24),
}


ASSET_DIR = os.path.join(os.path.dirname(__file__), "simenv", "pvz_sim", "assets")

PLANT_ASSETS = {
    "sunflower": "sunflower.png",
    "peashooter": "peashooter.png",
    "snowpea": "snowpea.png",
    "repeater": "repeater.png",
    "wallnut": "wallnut.png",
    "potatomine": "Potatomine.png",
    "squash": "squash.png",
    "cherrybomb": "cherrybomb.png",
    "spikeweed": "spikeweed.png",
    "kernelpult": "kernelpult.png",
    "melonpult": "melonpult.png",
}

ZOMBIE_ASSETS = {
    "zombie": "zombie.png",
    "zombie_cone": "zombie_cone.png",
    "zombie_bucket": "zombie_bucket.png",
    "zombie_flag": "zombie_flag.png",
}

PROJECTILE_ASSETS = {
    "pea": "pea.png",
    "kernel": "pea.png",
    "melon": "pea.png",
}


def main():
    env = SimPVZEnv()
    network = _load_network(env) if POLICY == "model" else None
    render_data, total_reward = _run_episode(env, network)
    print(f"Replay frames: {len(render_data)}, reward={total_reward:.2f}")
    render(render_data)


def _require_pygame():
    global pygame
    if pygame is not None:
        return pygame
    try:
        import pygame as pygame_module
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pygame is required for sim_game_render.py. "
            "Install project dependencies or run: "
            ".\\.venv\\Scripts\\python.exe -m pip install pygame"
        ) from exc
    pygame = pygame_module
    return pygame


def _load_network(env):
    if not MODEL_PATH:
        raise ValueError("MODEL_PATH must be set when POLICY = 'model'")
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    device = _resolve_device()
    network = DDQNNetwork(env, device=device)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    network.load_state_dict(state_dict)
    network.eval()
    return network


def _resolve_device():
    if DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return DEVICE


def _run_episode(env, network=None):
    env.enable_render_collection()
    state = transform_observation(env.reset())
    total_reward = 0.0
    done = False
    actions = 0

    while not done and actions < MAX_ACTIONS:
        action = _select_action(env, network, state)
        next_state, reward, done, _ = env.step(action)
        state = transform_observation(next_state)
        total_reward += float(reward)
        actions += 1

    return list(env.render_data), total_reward


def _select_action(env, network, state):
    if POLICY == "wait":
        return env.wait_action
    mask = env.mask_available_actions()
    if POLICY == "random":
        valid_actions = [idx for idx, valid in enumerate(mask) if valid]
        return random.choice(valid_actions)
    if POLICY == "model":
        return network.get_greedy_action(state, mask)
    raise ValueError(f"Unsupported POLICY: {POLICY}")


def render(render_info):
    pygame = _require_pygame()
    if not render_info:
        print("No render data to replay.")
        return

    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("calibri", 24)
    small_font = pygame.font.SysFont("calibri", 18)

    width = GRID_OFFSET_X * 2 + config.LANE_LENGTH * CELL_SIZE + 280
    height = GRID_OFFSET_Y + config.N_LANES * CELL_SIZE + INFO_HEIGHT
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("SimPVZ Agent Replay")

    sprites = _load_sprites()
    clock = pygame.time.Clock()

    frame_index = 0
    running = True
    while running and frame_index < len(render_info):
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        _draw_frame(screen, render_info[frame_index], sprites, font, small_font)
        pygame.display.flip()
        frame_index += 1

    pygame.quit()


def _load_sprites():
    sprites = {
        "plants": _load_sprite_group(PLANT_ASSETS, SPRITE_SCALE["plant"]),
        "zombies": {},
        "projectiles": _load_sprite_group(
            PROJECTILE_ASSETS, SPRITE_SCALE["projectile"]),
    }
    for key, filename in ZOMBIE_ASSETS.items():
        size = (
            SPRITE_SCALE["zombie_flag"]
            if key == "zombie_flag"
            else SPRITE_SCALE["zombie"]
        )
        sprites["zombies"][key] = _load_sprite(filename, size)
    return sprites


def _load_sprite_group(asset_map, size):
    return {
        key: _load_sprite(filename, size)
        for key, filename in asset_map.items()
    }


def _load_sprite(filename, size):
    pygame = _require_pygame()
    path = os.path.join(ASSET_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing render asset: {path}")
    image = pygame.image.load(path).convert_alpha()
    return pygame.transform.smoothscale(image, size)


def _draw_frame(screen, frame_info, sprites, font, small_font):
    screen.fill((128, 190, 92))
    _draw_grid(screen)
    _draw_objects(screen, frame_info, sprites)
    _draw_info(screen, frame_info, font, small_font)


def _draw_grid(screen):
    lawn_rect = pygame.Rect(
        GRID_OFFSET_X,
        GRID_OFFSET_Y,
        config.LANE_LENGTH * CELL_SIZE,
        config.N_LANES * CELL_SIZE,
    )
    pygame.draw.rect(screen, (102, 172, 77), lawn_rect)
    for col in range(config.LANE_LENGTH + 1):
        x = GRID_OFFSET_X + col * CELL_SIZE
        pygame.draw.line(
            screen,
            (78, 130, 58),
            (x, GRID_OFFSET_Y),
            (x, GRID_OFFSET_Y + config.N_LANES * CELL_SIZE),
            1,
        )
    for lane in range(config.N_LANES + 1):
        y = GRID_OFFSET_Y + lane * CELL_SIZE
        pygame.draw.line(
            screen,
            (78, 130, 58),
            (GRID_OFFSET_X, y),
            (GRID_OFFSET_X + config.LANE_LENGTH * CELL_SIZE, y),
            1,
        )


def _draw_objects(screen, frame_info, sprites):
    for lane in range(config.N_LANES):
        for plant_name, pos, hp in frame_info["plants"][lane]:
            key = _asset_key(plant_name)
            sprite = sprites["plants"].get(key)
            if sprite is None:
                continue
            x = GRID_OFFSET_X + pos * CELL_SIZE + 2
            y = GRID_OFFSET_Y + lane * CELL_SIZE + CELL_SIZE - sprite.get_height()
            screen.blit(sprite, (x, y))
            _draw_health_bar(screen, x, y - 6, sprite.get_width(), hp,
                             _plant_max_hp(plant_name))

        for zombie_name, pos, offset, hp in frame_info["zombies"][lane]:
            key = _asset_key(zombie_name)
            sprite = sprites["zombies"].get(key)
            if sprite is None:
                continue
            x = (
                GRID_OFFSET_X
                + CELL_SIZE * (pos + offset + 0.55)
                - sprite.get_width()
            )
            y = GRID_OFFSET_Y + lane * CELL_SIZE + CELL_SIZE - sprite.get_height()
            screen.blit(sprite, (x, y))
            _draw_health_bar(screen, x, y - 6, sprite.get_width(), hp,
                             _zombie_max_hp(zombie_name))

        for projectile_name, pos, offset in frame_info["projectiles"][lane]:
            key = PROJECTILE_ASSETS.get(_asset_key(projectile_name), "pea")
            sprite = sprites["projectiles"].get(_asset_key(projectile_name))
            if sprite is None:
                sprite = sprites["projectiles"].get("pea")
            if sprite is None or key is None:
                continue
            x = GRID_OFFSET_X + CELL_SIZE * (pos + offset + 0.5)
            y = GRID_OFFSET_Y + lane * CELL_SIZE + int(CELL_SIZE * 0.35)
            screen.blit(sprite, (x, y))


def _draw_health_bar(screen, x, y, width, hp, max_hp):
    if max_hp <= 0 or hp >= max_hp:
        return
    ratio = max(0.0, min(1.0, float(hp) / float(max_hp)))
    pygame.draw.rect(screen, (120, 0, 0), (x, y, width, 4))
    pygame.draw.rect(screen, (30, 200, 50), (x, y, int(width * ratio), 4))


def _draw_info(screen, frame_info, font, small_font):
    info_x = GRID_OFFSET_X + config.LANE_LENGTH * CELL_SIZE + 35
    info_y = GRID_OFFSET_Y
    lines = [
        f"Policy: {POLICY}",
        f"Time: {frame_info.get('time', 0)}s",
        f"Sun: {frame_info.get('sun', 0)}",
        f"Score: {frame_info.get('score', 0)}",
        f"Lives: {frame_info.get('lives', 1)}",
    ]
    for idx, line in enumerate(lines):
        screen.blit(font.render(line, True, (0, 0, 0)),
                    (info_x, info_y + idx * 28))

    cooldowns = frame_info.get("cooldowns", {})
    cd_y = info_y + 165
    screen.blit(font.render("Cooldowns", True, (0, 0, 0)), (info_x, cd_y))
    for idx, (name, value) in enumerate(cooldowns.items()):
        text = f"{name}: {value}s"
        screen.blit(small_font.render(text, True, (0, 0, 0)),
                    (info_x, cd_y + 28 + idx * 20))


def _asset_key(name):
    return str(name).replace("-", "").lower()


def _plant_max_hp(name):
    return {
        "Wallnut": 4000,
    }.get(str(name), 300)


def _zombie_max_hp(name):
    return {
        "Zombie": 270,
        "Zombie_flag": 270,
        "Zombie_cone": 640,
        "Zombie_bucket": 1370,
    }.get(str(name), 270)


if __name__ == "__main__":
    main()
