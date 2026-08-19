import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import { gameState } from './globals.js';

export class PhysicsSystem {
    static resolveCollision(entity) {
        // Resolve vs Walls
        const entityBox = new THREE.Box3().setFromObject(entity.mesh);
        // Shrink box slightly for gameplay feel (more forgiving)
        entityBox.expandByScalar(-0.1); 

        for (const wall of gameState.walls) {
            if (entityBox.intersectsBox(wall.boundingBox)) {
                // Determine Overlap
                const intersection = entityBox.clone().intersect(wall.boundingBox);
                const size = new THREE.Vector3();
                intersection.getSize(size);

                // Push out along smallest axis
                if (size.x < size.z) {
                    // Push X
                    const centerDiff = entity.mesh.position.x - wall.mesh.position.x;
                    const sign = Math.sign(centerDiff);
                    entity.mesh.position.x += (size.x + 0.01) * sign;
                    entity.velocity.x = 0;
                } else {
                    // Push Z
                    const centerDiff = entity.mesh.position.z - wall.mesh.position.z;
                    const sign = Math.sign(centerDiff);
                    entity.mesh.position.z += (size.z + 0.01) * sign;
                    entity.velocity.z = 0;
                }
                
                // Update box after push
                entityBox.setFromObject(entity.mesh);
                entityBox.expandByScalar(-0.1);
            }
        }
    }

    static checkProjectileCollisions(projectile) {
        // 1. Vs Walls
        const pPos = projectile.mesh.position;
        for (const wall of gameState.walls) {
            if (wall.boundingBox.containsPoint(pPos)) {
                return { hit: true, type: 'WALL', object: wall };
            }
        }

        // 2. Vs Entities (Enemies or Player)
        // Optimization: Simple distance check first
        const targets = projectile.ownerType === 'PLAYER' ? gameState.enemies : [gameState.player];
        
        for (const target of targets) {
            if (!target || target.isDead) continue;
            
            // Invulnerability during Dash
            if (target.isDashing) continue;

            const dist = pPos.distanceTo(target.mesh.position);
            if (dist < target.radius + 0.2) { // 0.2 projectile radius approx
                return { hit: true, type: 'ENTITY', object: target };
            }
        }

        return { hit: false };
    }
}