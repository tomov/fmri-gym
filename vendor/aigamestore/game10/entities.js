// entities.js - Game entities (doors, items, hiding spots)
import { gameState, TILE_SIZE } from './globals.js';

export class Door {
  constructor(x, y, w, h, locked = false, keyId = null) {
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    this.locked = locked;
    this.keyId = keyId;
    this.open = false;
  }

  interact() {
    if (this.locked) {
      // Check if player has key
      const hasKey = gameState.inventory.some(item => item.id === this.keyId);
      if (hasKey) {
        this.locked = false;
        this.open = true;
        gameState.score += 50;
        // Remove key from inventory
        gameState.inventory = gameState.inventory.filter(item => item.id !== this.keyId);
        return "unlocked";
      } else {
        return "locked";
      }
    } else {
      this.open = !this.open;
      return this.open ? "opened" : "closed";
    }
  }

  // Check if this door blocks movement (acts as wall when closed/locked)
  blocksMovement() {
    return !this.open;
  }

  render(p) {
    p.push();
    if (this.locked) {
      p.fill(139, 69, 19);
      p.stroke(100, 50, 10);
    } else if (this.open) {
      p.fill(101, 67, 33, 100); // Semi-transparent when open
      p.stroke(80, 50, 20);
    } else {
      p.fill(139, 90, 43);
      p.stroke(100, 70, 30);
    }
    p.strokeWeight(2);
    p.rect(this.x, this.y, this.w, this.h);
    
    // Door knob
    if (!this.open) {
      p.fill(180, 180, 100);
      p.noStroke();
      const knobX = this.w > this.h ? this.x + this.w * 0.8 : this.x + this.w / 2;
      const knobY = this.h > this.w ? this.y + this.h * 0.8 : this.y + this.h / 2;
      p.ellipse(knobX, knobY, 4, 4);
    }
    
    // Lock indicator
    if (this.locked) {
      p.fill(255, 215, 0);
      p.textSize(10);
      p.textAlign(p.CENTER, p.CENTER);
      p.text("🔒", this.x + this.w / 2, this.y + this.h / 2);
    }
    p.pop();
  }
}

export class Item {
  constructor(x, y, type, id, required = false) {
    this.x = x;
    this.y = y;
    this.type = type; // "key", "flashlight", "objective"
    this.id = id;
    this.required = required; // Whether this item is required to complete level
    this.collected = false;
    this.size = 12;
    this.glowPhase = 0;
  }

  update() {
    this.glowPhase += 0.1;
  }

  checkPickup() {
    if (this.collected || !gameState.player) return false;
    
    const dist = Math.hypot(this.x - gameState.player.x, this.y - gameState.player.y);
    return dist < 25;
  }

  collect() {
    this.collected = true;
    gameState.inventory.push({ type: this.type, id: this.id });
    
    // Track required items
    if (this.required && gameState.requiredItemIds.includes(this.id)) {
      gameState.collectedRequiredItems.add(this.id);
    }
    
    // Add visual feedback
    if (typeof window.addMessage === 'function') {
      if (this.type === "key") {
        gameState.score += 20;
        window.addMessage("Key collected! +20", "success");
      } else if (this.type === "flashlight") {
        gameState.hasFlashlight = true;
        gameState.score += 20;
        window.addMessage("Flashlight collected! +20", "success");
      } else if (this.type === "objective") {
        gameState.score += 30;
        const requiredMsg = this.required ? " (Required!)" : "";
        window.addMessage("Objective collected! +30" + requiredMsg, "success");
      }
      
      // Check if all required items collected
      if (this.required && gameState.collectedRequiredItems.size === gameState.requiredItemIds.length) {
        window.addMessage("All objectives collected! Exit unlocked!", "success");
        gameState.score += 50;
      }
    }
  }

  render(p) {
    if (this.collected) return;
    
    p.push();
    const glow = 150 + Math.sin(this.glowPhase) * 50;
    
    // Extra glow for required items
    if (this.required) {
      p.fill(255, 100, 255, glow * 0.5);
      p.noStroke();
      p.ellipse(this.x, this.y, this.size + 10, this.size + 10);
    }
    
    if (this.type === "key") {
      p.fill(255, 215, 0, glow);
      p.noStroke();
      p.ellipse(this.x, this.y, this.size + 4, this.size + 4);
      p.fill(255, 215, 0);
      p.stroke(200, 170, 0);
      p.strokeWeight(2);
      p.rect(this.x - 4, this.y - 2, 8, 4);
      p.rect(this.x + 2, this.y - 6, 2, 12);
    } else if (this.type === "flashlight") {
      p.fill(200, 200, 200, glow);
      p.noStroke();
      p.ellipse(this.x, this.y, this.size + 4, this.size + 4);
      p.fill(150, 150, 150);
      p.stroke(100);
      p.strokeWeight(2);
      p.rect(this.x - 4, this.y - 6, 8, 12);
      p.fill(255, 255, 200);
      p.noStroke();
      p.rect(this.x - 4, this.y - 6, 8, 4);
    } else {
      p.fill(100, 200, 255, glow);
      p.noStroke();
      p.ellipse(this.x, this.y, this.size + 4, this.size + 4);
      p.fill(100, 200, 255);
      p.stroke(50, 150, 200);
      p.strokeWeight(2);
      p.ellipse(this.x, this.y, this.size, this.size);
      
      // Star indicator for required items
      if (this.required) {
        p.fill(255, 255, 100);
        p.noStroke();
        p.textSize(10);
        p.textAlign(p.CENTER, p.CENTER);
        p.text("★", this.x, this.y - 15);
      }
    }
    p.pop();
  }
}

export class HidingSpot {
  constructor(x, y, w, h, type = "closet") {
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    this.type = type;
    this.occupied = false;
  }

  canHide() {
    if (!gameState.player) return false;
    const dist = Math.hypot(
      this.x + this.w / 2 - gameState.player.x,
      this.y + this.h / 2 - gameState.player.y
    );
    return dist < 30;
  }

  render(p) {
    p.push();
    if (this.type === "closet") {
      p.fill(101, 67, 33);
      p.stroke(70, 45, 20);
      p.strokeWeight(2);
      p.rect(this.x, this.y, this.w, this.h);
      
      // Doors
      p.stroke(60, 40, 15);
      p.line(this.x + this.w / 2, this.y, this.x + this.w / 2, this.y + this.h);
    } else if (this.type === "table") {
      p.fill(139, 90, 43);
      p.stroke(100, 70, 30);
      p.strokeWeight(2);
      p.rect(this.x, this.y, this.w, this.h);
      
      // Legs
      p.stroke(80, 60, 25);
      p.strokeWeight(3);
      p.line(this.x + 5, this.y + this.h, this.x + 5, this.y + this.h + 10);
      p.line(this.x + this.w - 5, this.y + this.h, this.x + this.w - 5, this.y + this.h + 10);
    }
    p.pop();
  }
}

export class ExitZone {
  constructor(x, y, w, h) {
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    this.lastLockedMsgTime = 0;
  }

  isUnlocked() {
    // Check if all required items have been collected
    return gameState.collectedRequiredItems.size >= gameState.requiredItemIds.length;
  }

  checkPlayer() {
    if (!gameState.player) return false;
    
    const inZone = gameState.player.x > this.x &&
                   gameState.player.x < this.x + this.w &&
                   gameState.player.y > this.y &&
                   gameState.player.y < this.y + this.h;
    
    // Show message if player tries to exit without required items
    if (inZone && !this.isUnlocked()) {
      const now = Date.now();
      // Only show message every 2 seconds to prevent spam
      if (typeof window.addMessage === 'function' && (now - this.lastLockedMsgTime > 2000)) {
        const remaining = gameState.requiredItemIds.length - gameState.collectedRequiredItems.size;
        window.addMessage(`Exit locked! Collect ${remaining} more objective(s)`, "warning");
        this.lastLockedMsgTime = now;
      }
      return false;
    }
    
    return inZone && this.isUnlocked();
  }

  render(p) {
    p.push();
    
    const unlocked = this.isUnlocked();
    
    if (unlocked) {
      // Unlocked - green and pulsing
      const pulse = 100 + Math.sin(p.frameCount * 0.1) * 50;
      p.fill(100, 255, 100, pulse);
      p.stroke(50, 200, 50);
      p.strokeWeight(3);
      p.rect(this.x, this.y, this.w, this.h);
      p.fill(255);
      p.noStroke();
      p.textAlign(p.CENTER, p.CENTER);
      p.textSize(14);
      p.text("EXIT", this.x + this.w / 2, this.y + this.h / 2 - 5);
      p.textSize(10);
      p.text("✓ UNLOCKED", this.x + this.w / 2, this.y + this.h / 2 + 8);
    } else {
      // Locked - red/gray and darker
      p.fill(150, 80, 80, 150);
      p.stroke(120, 60, 60);
      p.strokeWeight(3);
      p.rect(this.x, this.y, this.w, this.h);
      p.fill(200, 100, 100);
      p.noStroke();
      p.textAlign(p.CENTER, p.CENTER);
      p.textSize(14);
      p.text("EXIT", this.x + this.w / 2, this.y + this.h / 2 - 5);
      p.textSize(10);
      const remaining = gameState.requiredItemIds.length - gameState.collectedRequiredItems.size;
      p.text(`🔒 ${remaining} items needed`, this.x + this.w / 2, this.y + this.h / 2 + 8);
    }
    
    p.pop();
  }
}