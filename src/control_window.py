import carla
import pygame

client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
world = client.get_world()
world.wait_for_tick()
actor_list = world.get_actors().filter(
    '*model3*'
)
vehicle_list = []
for vehicle in actor_list:
    vehicle_list.append(vehicle)

vehicle = vehicle_list[0]

pygame.init()

size = (512, 256)
pygame.display.set_caption("Pygame carla manual control window")
screen = pygame.display.set_mode(size)

control = carla.VehicleControl()
clock = pygame.time.Clock()
done = False

while not done:
    keys = pygame.key.get_pressed()

    if keys[pygame.K_UP] or keys[pygame.K_w]:
        control.throttle = min(control.throttle + 0.05, 1.0)
    else:
        control.throttle = 0.0

    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        control.brake = min(control.brake + 0.2, 1.0)
    else:
        control.brake = 0.0

    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        control.steer = max(control.steer - 0.05, -1.0)
    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        control.steer = min(control.steer + 0.05, 1.0)
    else:
        control.steer = 0.0

    control.hand_brake = keys[pygame.K_SPACE]

    vehicle.apply_control(control)
    world.tick()

    pygame.display.flip()
    pygame.display.update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    clock.tick(60)
