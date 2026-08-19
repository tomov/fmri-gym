import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import { gameState } from './globals.js';

export function setupLighting() {
    // Ambient - Dark mood
    const ambient = new THREE.AmbientLight(0xffffff, 0.3);
    gameState.scene.add(ambient);
    gameState.ambientLight = ambient;
    
    // Directional - Moon/Street light
    const dirLight = new THREE.DirectionalLight(0xaaccff, 0.8);
    dirLight.position.set(-10, 20, -10);
    dirLight.castShadow = true;
    
    // Shadow config
    dirLight.shadow.mapSize.width = 1024;
    dirLight.shadow.mapSize.height = 1024;
    dirLight.shadow.camera.left = -30;
    dirLight.shadow.camera.right = 30;
    dirLight.shadow.camera.top = 30;
    dirLight.shadow.camera.bottom = -30;
    
    gameState.scene.add(dirLight);
    gameState.lights.push(dirLight);
    
    // Maybe a spotlight following player?
    const spot = new THREE.SpotLight(0xffaa00, 0.5);
    spot.position.set(0, 15, 0);
    spot.angle = Math.PI / 6;
    spot.penumbra = 0.5;
    spot.castShadow = false;
    gameState.scene.add(spot);
    gameState.playerSpot = spot;
}

export function updateLighting() {
    if (gameState.player && gameState.playerSpot) {
        gameState.playerSpot.position.x = gameState.player.mesh.position.x;
        gameState.playerSpot.position.z = gameState.player.mesh.position.z + 5;
        gameState.playerSpot.target.position.copy(gameState.player.mesh.position);
        gameState.playerSpot.target.updateMatrixWorld();
    }
}