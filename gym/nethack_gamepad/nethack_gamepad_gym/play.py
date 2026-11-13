"""Play NetHack with a gamepad standalone, without fmri-gym experiment recording."""

from __future__ import annotations

import argparse


def main() -> None:
    """Launch with native character selection and persistent upstream save files."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="For recorded runs, use fmri-play with --subject, --ses, --run, --rig and --curriculum.")
    parser.add_argument("--seed", type=int, default=None,
                        help="isolated reproducible episode; fixes the game calendar to 2021-01-01")
    parser.add_argument("--mode", choices=("classic", "modern"), default="modern")
    parser.add_argument("--campaign", type=str, help="replay checkpoint file to resume and save on exit")
    args = parser.parse_args()
    import pygame
    from pygame._sdl2 import controller as sdl_controller
    from .build import default_home
    from .env import NetHackGamepad

    from pathlib import Path
    from .campaign import save_campaign
    campaign = Path(args.campaign).expanduser() if args.campaign else None
    initial = campaign.read_text().strip() if campaign and campaign.exists() else None
    player = None if args.seed is not None or campaign else str(default_home() / ("player-" + args.mode))
    environment = NetHackGamepad(player_directory=player, game_mode=args.mode, initial_state=initial)
    engine = environment.unwrapped
    pygame.display.init()
    screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
    pygame.display.set_caption("NetHack — native desktop controller port")
    pygame.joystick.init()
    sdl_controller.init()
    controllers = {}
    def attach(index):
        if sdl_controller.is_controller(index):
            handle = sdl_controller.Controller(index)
            controllers[handle.id] = handle
    for index in range(pygame.joystick.get_count()):
        attach(index)
    buttons = {pygame.CONTROLLER_BUTTON_DPAD_UP: 0,
               pygame.CONTROLLER_BUTTON_DPAD_RIGHT: 1,
               pygame.CONTROLLER_BUTTON_DPAD_DOWN: 2,
               pygame.CONTROLLER_BUTTON_DPAD_LEFT: 3,
               pygame.CONTROLLER_BUTTON_A: 4, pygame.CONTROLLER_BUTTON_B: 5,
               pygame.CONTROLLER_BUTTON_X: 6, pygame.CONTROLLER_BUTTON_Y: 7}
    bindings = {pygame.K_UP: 0, pygame.K_RIGHT: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3,
                pygame.K_z: 4, pygame.K_x: 5, pygame.K_c: 6, pygame.K_v: 7}
    try:
        environment.reset(seed=args.seed)
        running = True
        clock = pygame.time.Clock()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.CONTROLLERDEVICEADDED:
                    attach(event.device_index)
                elif event.type == pygame.CONTROLLERDEVICEREMOVED:
                    handle = controllers.pop(event.instance_id, None)
                    if handle and handle.get_init():
                        handle.quit()
                elif event.type == pygame.CONTROLLERBUTTONDOWN and event.button in buttons:
                    if not engine.session.finished:
                        environment.step(buttons[event.button])
                    elif buttons[event.button] == 4:
                        environment.initial_state = None
                        environment.reset(seed=args.seed)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in bindings and not engine.session.finished:
                        environment.step(bindings[event.key])
                    elif event.key == pygame.K_z and engine.session.finished:
                        environment.initial_state = None
                        environment.reset(seed=args.seed)
            pixels = environment.render()
            surface = pygame.surfarray.make_surface(pixels.transpose(1, 0, 2))
            width, height = screen.get_size()
            scale = min(width / surface.get_width(), height / surface.get_height())
            surface = pygame.transform.scale(surface,
                         (int(surface.get_width() * scale), int(surface.get_height() * scale)))
            screen.fill((0, 0, 0))
            screen.blit(surface, surface.get_rect(center=(width // 2, height // 2)))
            pygame.display.flip()
            clock.tick(30)
    finally:
        try:
            if campaign and environment._started:
                save_campaign(campaign, environment.get_continuation_state())
        finally:
            environment.close()
            for handle in controllers.values():
                if handle.get_init():
                    handle.quit()
            sdl_controller.quit()
            pygame.display.quit()



if __name__ == "__main__":
    main()
