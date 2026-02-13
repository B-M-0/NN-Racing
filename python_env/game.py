import pygame
import math
import sys
import os

pygame.init()

# ----------------- Configuration -----------------
track_image = pygame.image.load("track.png")
TRACK_WIDTH, TRACK_HEIGHT = track_image.get_size()
screen = pygame.display.set_mode((TRACK_WIDTH, TRACK_HEIGHT))
clock = pygame.time.Clock()

# Fonts
timer_font = pygame.font.SysFont("Courier", 30, bold=True) 
font = pygame.font.SysFont("Arial", 16, bold=True)
inst_font = pygame.font.SysFont("Arial", 14)

wall_mask = pygame.mask.from_threshold(track_image, (0, 0, 0, 255), (20, 20, 20, 255))
SAVE_FILE = "track_data.txt"
ORANGE = (255, 165, 0)

def format_time(ms):
    if ms == float('inf') or ms < 0: return "--:--:--"
    seconds = (ms // 1000) % 60
    minutes = (ms // 60000)
    hundredths = (ms // 10) % 100
    return f"{minutes:02}:{seconds:02}:{hundredths:02}"

class Car:
    def __init__(self, x, y):
        self.spawn_pos = [x, y]
        self.width, self.height = 20, 10
        self.reset()

        # Create Car Surface
        self.original_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.rect(self.original_surf, (200, 0, 0), (0, 0, self.width, self.height))
        # Updated Headlight: Orange
        pygame.draw.rect(self.original_surf, ORANGE, (self.width-3, 0, 3, self.height))

    def reset(self):
        self.x, self.y = self.spawn_pos
       
        self.speed = 0
        self.angle = 0

        self.current_cp_idx = 0 
        self.laps = 0
        self.lap_start_time = pygame.time.get_ticks()
        self.last_lap_time  = 0
        self.best_lap_time  = float('inf')

    def update(self, keys, checkpoints):
        old_x, old_y = self.x, self.y
        old_angle    = self.angle

        if keys[pygame.K_UP]:
            self.speed += 0.15
        elif keys[pygame.K_DOWN]: 
            self.speed -= 0.15
        else: 
            self.speed *= 0.96
        
        self.speed = max(-2, min(self.speed, 5))
        if self.speed != 0:
            dir_mult = 1 if self.speed > 0 else -1
            if keys[pygame.K_LEFT]: self.angle += 5 * dir_mult
            if keys[pygame.K_RIGHT]: self.angle -= 5 * dir_mult

        self.x += math.cos(math.radians(self.angle)) * self.speed
        self.y -= math.sin(math.radians(self.angle)) * self.speed

        rotated = pygame.transform.rotate(self.original_surf, self.angle)
        rect = rotated.get_rect(center=(int(self.x), int(self.y)))
        car_mask = pygame.mask.from_surface(rotated)
        
        if wall_mask.overlap(car_mask, (rect.x, rect.y)):
            self.x, self.y = old_x, old_y
            self.angle     = old_angle
            self.speed    *= -0.5

        if checkpoints:
            p1, p2 = checkpoints[self.current_cp_idx]
            dist   = self.dist_point_to_line(pygame.Vector2(self.x, self.y), pygame.Vector2(p1), pygame.Vector2(p2))
           
            if dist < 15:
                self.current_cp_idx = (self.current_cp_idx + 1) % len(checkpoints)
                if self.current_cp_idx == 0:
                    now = pygame.time.get_ticks()
                    self.last_lap_time = now - self.lap_start_time
                    if self.last_lap_time < self.best_lap_time:
                        self.best_lap_time = self.last_lap_time
                    self.lap_start_time = now
                    self.laps += 1

    def dist_point_to_line(self, p, a, b):
        pa = p - a
        ba = b - a
        denom = ba.dot(ba)
        if denom == 0: return pa.length()
        t = max(0, min(1, pa.dot(ba) / denom))
        return (p - (a + t * ba)).length()

    def draw(self, surf):
        rotated = pygame.transform.rotate(self.original_surf, self.angle)
        rect = rotated.get_rect(center=(int(self.x), int(self.y)))
        surf.blit(rotated, rect.topleft)

# ----------------- Persistence -----------------
def save_data(spawn, checkpoints):
    with open(SAVE_FILE, "w") as f:
        f.write(f"{spawn[0]},{spawn[1]}\n")
        for cp in checkpoints:
            f.write(f"{cp[0][0]},{cp[0][1]}|{cp[1][0]},{cp[1][1]}\n")

def load_data():
    if not os.path.exists(SAVE_FILE): return [TRACK_WIDTH//2, TRACK_HEIGHT//2], []
    try:
        # prettify thihs string manipulation
        with open(SAVE_FILE, "r") as f:
            lines = f.readlines()
            spawn = [int(i) for i in lines[0].strip().split(",")]
            checkpoints = []
            for line in lines[1:]:
                p1_s, p2_s = line.strip().split("|")
                p1 = [int(i) for i in p1_s.split(",")]
                p2 = [int(i) for i in p2_s.split(",")]
                checkpoints.append((p1, p2))
            return spawn, checkpoints
    except: return [TRACK_WIDTH//2, TRACK_HEIGHT//2], []



# ---------------- Rendering Settings ----------
render = True
# Modes: all, minmax, top10, best
render_mode = 'all'
render_interval = 10
tick_rate = 60

# ----------------- Main Loop -----------------
spawn_pos, user_checkpoints = load_data()
car = Car(spawn_pos[0], spawn_pos[1])
temp_point = None





while True:
    screen.blit(track_image, (0, 0))
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); sys.exit()
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: # Left Click
                if temp_point is None: temp_point = event.pos
                else:
                    user_checkpoints.append((temp_point, event.pos))
                    temp_point = None
                    save_data(car.spawn_pos, user_checkpoints)
            if event.button == 3: # Right Click
                car.spawn_pos = list(event.pos)
                car.reset()
                save_data(car.spawn_pos, user_checkpoints)
        
        if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
            user_checkpoints = []
            if os.path.exists(SAVE_FILE): os.remove(SAVE_FILE)
            car.reset()

    car.update(pygame.key.get_pressed(), user_checkpoints)

    # Draw Checkpoints and Numbers
    for i, cp in enumerate(user_checkpoints):
        color = (0, 255, 0) if i == car.current_cp_idx else (255, 255, 0)
        pygame.draw.line(screen, color, cp[0], cp[1], 3)
        # Display Number
        num_txt = font.render(str(i+1), True, color)
        screen.blit(num_txt, (cp[0][0], cp[0][1] - 20))

    if temp_point:
        pygame.draw.line(screen, (200, 200, 200), temp_point, pygame.mouse.get_pos(), 1)

    # --- UI ---
    curr_ms = pygame.time.get_ticks() - car.lap_start_time
    screen.blit(timer_font.render(format_time(curr_ms), True, (255, 255, 255)), (TRACK_WIDTH - 180, 20))
    screen.blit(font.render(f"BEST: {format_time(car.best_lap_time)}", True, (0, 255, 255)), (TRACK_WIDTH - 180, 60))
    screen.blit(font.render(f"LAST: {format_time(car.last_lap_time)}", True, (200, 200, 200)), (TRACK_WIDTH - 180, 85))
    
    # Instructions
    controls = ["L-Click x2: New Gate", "R-Click: Set Spawn", "C Key: Clear All"]
    for i, line in enumerate(controls):
        screen.blit(inst_font.render(line, True, (255, 255, 255)), (15, 15 + (i * 18)))

    car.draw(screen)
    pygame.display.flip()
    clock.tick(tick_rate)