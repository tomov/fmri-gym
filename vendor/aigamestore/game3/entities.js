import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import { gameState, CONFIG, logGameEvent } from './globals.js';
import { getInputVector, getShootVector } from './input.js';
import { PhysicsSystem } from './physics.js';
import { spawnParticles } from './particles.js';
import { raycastScene, randomRange } from './utils.js';

class Entity {
    constructor(x, y, z) {
        this.mesh = new THREE.Group();
        this.mesh.position.set(x, y, z);
        this.velocity = new THREE.Vector3();
        this.isDead = false;
        this.radius = 0.5;
        gameState.scene.add(this.mesh);
    }
    
    update(dt) {
        // Base physics integration
        this.mesh.position.add(this.velocity.clone().multiplyScalar(dt));
    }

    takeDamage(amount) {
        // Implementation in subclasses
    }
    
    destroy() {
        this.isDead = true;
        gameState.scene.remove(this.mesh);
    }
}

export class Player extends Entity {
    constructor(x, z) {
        super(x, 1, z);
        
        // Body
        const geometry = new THREE.BoxGeometry(0.8, 1.8, 0.8);
        const material = new THREE.MeshStandardMaterial({ 
            color: CONFIG.COLORS.PLAYER, 
            roughness: 0.2,
            metalness: 0.5 
        });
        this.bodyMesh = new THREE.Mesh(geometry, material);
        this.bodyMesh.castShadow = true;
        this.bodyMesh.position.y = 0; // Center pivot
        this.mesh.add(this.bodyMesh);
        
        // Stats
        this.health = CONFIG.PLAYER_MAX_HEALTH;
        this.speed = CONFIG.PLAYER_SPEED;
        
        // State
        this.isDashing = false;
        this.dashTimer = 0;
        this.dashCooldownTimer = 0;
        
        // Weapons
        this.weapons = ['PISTOL', 'SHOTGUN', 'RIFLE'];
        this.weaponIndex = 0;
        this.currentWeapon = this.weapons[0];
        this.weaponCooldown = 0;
        this.swapCooldown = 0;
        
        this.radius = 0.4;
    }
    
    update(dt) {
        if (this.isDead) return;

        // --- DASH LOGIC ---
        if (this.isDashing) {
            this.dashTimer -= dt;
            if (this.dashTimer <= 0) {
                this.isDashing = false;
                this.dashCooldownTimer = CONFIG.PLAYER_DASH_COOLDOWN;
            }
            // Continue movement at high speed
            super.update(dt * 3); 
            PhysicsSystem.resolveCollision(this);
            return; // Skip normal input
        }

        if (this.dashCooldownTimer > 0) this.dashCooldownTimer -= dt;

        // --- WEAPON SWAP ---
        if (this.swapCooldown > 0) this.swapCooldown -= dt;
        if (gameState.input.swap && this.swapCooldown <= 0) {
            this.weaponIndex = (this.weaponIndex + 1) % this.weapons.length;
            this.currentWeapon = this.weapons[this.weaponIndex];
            this.swapCooldown = 0.3; // Prevent rapid cycling
            logGameEvent('WEAPON_SWAP', { weapon: this.currentWeapon });
        }

        // --- MOVEMENT ---
        const inputVec = getInputVector();
        const moveDir = new THREE.Vector3(inputVec.x, 0, inputVec.z).normalize();
        
        let currentSpeed = this.speed;
        if (gameState.input.walk) currentSpeed *= 0.5;
        
        // Dash Activation
        if (gameState.input.dash && this.dashCooldownTimer <= 0 && moveDir.length() > 0) {
            this.isDashing = true;
            this.dashTimer = CONFIG.PLAYER_DASH_DURATION;
            this.velocity.copy(moveDir).multiplyScalar(CONFIG.PLAYER_DASH_SPEED * 60); 
        }

        // Apply Movement (Acceleration/Friction model)
        const targetVel = moveDir.multiplyScalar(10); // 10 units/sec walking speed approx
        
        // Instant response for tight controls
        this.velocity.x = targetVel.x;
        this.velocity.z = targetVel.z;
        
        super.update(dt);
        PhysicsSystem.resolveCollision(this);
        
        // --- SHOOTING ---
        if (this.weaponCooldown > 0) this.weaponCooldown -= dt;
        
        const shootVec = getShootVector();
        const shootDir = new THREE.Vector3(shootVec.x, 0, shootVec.z);
        
        if (shootDir.lengthSq() > 0.1) {
            shootDir.normalize();
            // Face shooting direction
            const angle = Math.atan2(shootDir.x, shootDir.z);
            this.mesh.rotation.y = angle;
            
            if (this.weaponCooldown <= 0) {
                this.fireWeapon(shootDir);
            }
        } else if (moveDir.lengthSq() > 0.1) {
            // Face movement if not shooting
            const angle = Math.atan2(moveDir.x, moveDir.z);
            this.mesh.rotation.y = angle;
        }

        // Logging position for testing
        if (gameState.frameCount % 60 === 0) {
            logGameEvent('PLAYER_POS', { x: this.mesh.position.x, z: this.mesh.position.z });
        }
    }
    
    fireWeapon(direction) {
        const pos = this.mesh.position.clone().add(direction.clone().multiplyScalar(0.6));
        
        if (this.currentWeapon === 'PISTOL') {
            const p = new Projectile(pos, direction, 'PLAYER', 15);
            gameState.projectiles.push(p);
            this.weaponCooldown = 0.25;
            
        } else if (this.currentWeapon === 'SHOTGUN') {
            // Fire 5 pellets
            for (let i = 0; i < 5; i++) {
                const spreadAngle = randomRange(-0.2, 0.2);
                // Rotate direction around Y axis
                const spreadDir = direction.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), spreadAngle);
                const p = new Projectile(pos, spreadDir, 'PLAYER', 8);
                p.lifeTime = 0.6; // Short range
                gameState.projectiles.push(p);
            }
            this.weaponCooldown = 0.8;
            
        } else if (this.currentWeapon === 'RIFLE') {
            const p = new Projectile(pos, direction, 'PLAYER', 10);
            p.speed = 30; // Faster bullets
            p.velocity.copy(p.direction).multiplyScalar(p.speed);
            gameState.projectiles.push(p);
            this.weaponCooldown = 0.1;
        }
    }

    takeDamage(amount) {
        if (this.isDashing || this.isDead) return;
        
        this.health -= amount;
        spawnParticles(this.mesh.position, 'BLOOD', 3);
        
        if (this.health <= 0) {
            this.health = 0;
            this.die();
        }
    }
    
    die() {
        this.isDead = true;
        this.mesh.visible = false;
        spawnParticles(this.mesh.position, 'BLOOD', 20);
        logGameEvent('PLAYER_DEATH', {});
        gameState.gamePhase = "GAME_OVER_LOSE";
    }
}

export class Enemy extends Entity {
    constructor(x, z, type = 'GRUNT') {
        super(x, 1, z);
        this.type = type; // GRUNT (Melee), SHOOTER (Ranged)
        
        const color = type === 'GRUNT' ? 0xCC0000 : 0xFF4400;
        const geo = new THREE.BoxGeometry(0.8, 1.8, 0.8);
        const mat = new THREE.MeshStandardMaterial({ color: color });
        this.bodyMesh = new THREE.Mesh(geo, mat);
        this.bodyMesh.castShadow = true;
        this.mesh.add(this.bodyMesh);
        
        this.health = 30;
        this.state = 'IDLE'; // IDLE, CHASE, ATTACK, SEARCH
        this.stateTimer = 0;
        this.attackCooldown = 0;
        this.detectionRange = 15;
        this.attackRange = type === 'GRUNT' ? 1.5 : 10;
        
        this.navDir = new THREE.Vector3();
    }
    
    update(dt) {
        if (this.isDead || !gameState.player) return;

        if (this.attackCooldown > 0) this.attackCooldown -= dt;

        const distToPlayer = this.mesh.position.distanceTo(gameState.player.mesh.position);
        
        // AI State Machine
        switch (this.state) {
            case 'IDLE':
                if (distToPlayer < this.detectionRange) {
                    // Check Line of Sight
                    const dirToPlayer = new THREE.Vector3().subVectors(gameState.player.mesh.position, this.mesh.position).normalize();
                    const hit = raycastScene(this.mesh.position, dirToPlayer, this.detectionRange, [this]);
                    
                    if (hit && hit.type === 'PLAYER') {
                        this.state = 'CHASE';
                        logGameEvent('ENEMY_ALERT', { id: this.mesh.id });
                    }
                }
                break;
                
            case 'CHASE':
                if (distToPlayer > this.detectionRange * 1.5) {
                    this.state = 'IDLE';
                } else if (distToPlayer <= this.attackRange) {
                    // Check LoS again before attacking
                    const dir = new THREE.Vector3().subVectors(gameState.player.mesh.position, this.mesh.position).normalize();
                    const hit = raycastScene(this.mesh.position, dir, this.attackRange, [this]);
                    if (hit && hit.type === 'PLAYER') {
                        this.state = 'ATTACK';
                    } else {
                        // Move to sight
                        this.moveToPlayer(dt);
                    }
                } else {
                    this.moveToPlayer(dt);
                }
                break;
                
            case 'ATTACK':
                if (distToPlayer > this.attackRange + 1) {
                    this.state = 'CHASE';
                } else {
                    // Face player
                    this.mesh.lookAt(gameState.player.mesh.position.x, this.mesh.position.y, gameState.player.mesh.position.z);
                    
                    if (this.attackCooldown <= 0) {
                        this.performAttack();
                        this.attackCooldown = this.type === 'GRUNT' ? 1.0 : 2.0;
                    }
                }
                break;
        }
        
        super.update(dt);
        PhysicsSystem.resolveCollision(this);
    }
    
    moveToPlayer(dt) {
        const dir = new THREE.Vector3().subVectors(gameState.player.mesh.position, this.mesh.position).normalize();
        this.mesh.lookAt(gameState.player.mesh.position.x, this.mesh.position.y, gameState.player.mesh.position.z);
        
        const speed = this.type === 'GRUNT' ? 6 : 4; // Units per sec
        
        this.velocity.copy(dir).multiplyScalar(speed);
        
        // Very basic avoidance (if hitting wall, stop)
        // Handled by PhysicsSystem.resolveCollision call in update
    }
    
    performAttack() {
        if (this.type === 'GRUNT') {
            // Melee Hit
            if (this.mesh.position.distanceTo(gameState.player.mesh.position) < 2.0) {
                gameState.player.takeDamage(10);
            }
        } else {
            // Shoot
            const dir = new THREE.Vector3().subVectors(gameState.player.mesh.position, this.mesh.position).normalize();
            const p = new Projectile(
                this.mesh.position.clone().add(dir.multiplyScalar(0.6)),
                dir,
                'ENEMY',
                10 // Damage
            );
            gameState.projectiles.push(p);
        }
    }
    
    takeDamage(amount) {
        if (this.isDead) return;
        this.health -= amount;
        this.state = 'CHASE'; // Alert on damage
        spawnParticles(this.mesh.position, 'BLOOD', 3);
        
        // Knockback
        // this.velocity.add(dir.multiplyScalar(5)); // Need source dir
        
        if (this.health <= 0) {
            this.die();
        }
    }
    
    die() {
        this.isDead = true;
        gameState.scene.remove(this.mesh);
        spawnParticles(this.mesh.position, 'BLOOD', 15);
        gameState.score += 100;
        
        // Remove from list
        const idx = gameState.enemies.indexOf(this);
        if (idx > -1) gameState.enemies.splice(idx, 1);
        
        logGameEvent('ENEMY_KILLED', { remaining: gameState.enemies.length });
    }
}

export class Projectile extends Entity {
    constructor(pos, dir, ownerType, damage = 10) {
        super(pos.x, pos.y, pos.z);
        
        const geo = new THREE.SphereGeometry(0.2, 8, 8);
        const mat = new THREE.MeshBasicMaterial({ color: CONFIG.COLORS.PROJECTILE });
        this.mesh = new THREE.Mesh(geo, mat);
        this.mesh.position.copy(pos);
        gameState.scene.add(this.mesh); // Re-add specific mesh
        
        this.direction = dir.normalize();
        this.speed = 20; // Fast
        this.ownerType = ownerType; // PLAYER or ENEMY
        this.damage = damage;
        this.lifeTime = 2.0;
        
        // Correct velocity setup
        this.velocity.copy(this.direction).multiplyScalar(this.speed);
    }
    
    update(dt) {
        this.lifeTime -= dt;
        if (this.lifeTime <= 0) {
            this.destroy();
            return;
        }
        
        // Move manually to check collision raycast style for high speed?
        // For now, simple movement step + collision check
        const nextPos = this.mesh.position.clone().add(this.velocity.clone().multiplyScalar(dt));
        
        // Collision Detection
        const hitData = PhysicsSystem.checkProjectileCollisions(this);
        if (hitData.hit) {
            this.handleHit(hitData);
            return; // Stop moving
        }
        
        this.mesh.position.copy(nextPos);
    }
    
    handleHit(data) {
        if (data.type === 'WALL') {
            spawnParticles(this.mesh.position, 'WALL_HIT', 5);
        } else if (data.type === 'ENTITY') {
            data.object.takeDamage(this.damage);
        }
        this.destroy();
    }
    
    destroy() {
        if (this.isDead) return;
        this.isDead = true;
        gameState.scene.remove(this.mesh);
        
        const idx = gameState.projectiles.indexOf(this);
        if (idx > -1) gameState.projectiles.splice(idx, 1);
    }
}

export class Wall {
    constructor(x, z, width, depth) {
        this.width = width;
        this.depth = depth;
        
        const h = 3;
        const geo = new THREE.BoxGeometry(width, h, depth);
        const mat = new THREE.MeshStandardMaterial({ 
            color: CONFIG.COLORS.WALL,
            roughness: 0.8
        });
        this.mesh = new THREE.Mesh(geo, mat);
        this.mesh.position.set(x, h/2, z);
        this.mesh.castShadow = true;
        this.mesh.receiveShadow = true;
        
        gameState.scene.add(this.mesh);
        
        // Physics Box
        this.boundingBox = new THREE.Box3().setFromObject(this.mesh);
        gameState.walls.push(this);
    }
}