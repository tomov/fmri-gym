// tube.js - Tube class and related functions

import { COLOR_MAP } from './globals.js';

export class Tube {
  constructor(capacity, colors, x, y, width, p) {
    this.capacity = capacity;
    this.colors = [...colors];
    this.x = x;
    this.y = y;
    this.width = width;
    this.p = p;
    
    // Calculate dimensions
    this.totalHeight = 200;
    this.layerHeight = this.totalHeight / this.capacity;
  }
  
  draw(isHighlighted, isSelected) {
    const p = this.p;
    
    // Calculate visual offset - lift selected tube
    const yOffset = isSelected ? -20 : 0;
    const drawY = this.y + yOffset;
    
    // Draw indicator arrow above tube
    if (isHighlighted || isSelected) {
      p.push();
      p.noStroke();
      if (isSelected) {
        p.fill(255, 220, 0); // Yellow for selected
      } else {
        p.fill(100, 200, 255); // Blue for highlighted
      }
      
      // Downward pointing triangle
      const arrowY = drawY - 15;
      p.triangle(
        this.x + this.width / 2 - 8, arrowY,
        this.x + this.width / 2 + 8, arrowY,
        this.x + this.width / 2, arrowY + 10
      );
      p.pop();
    }
    
    // Draw tube outline
    p.push();
    p.noFill();
    
    if (isSelected) {
      p.strokeWeight(4);
      p.stroke(255, 220, 0);
      // Glow effect
      if (p.drawingContext) {
        p.drawingContext.shadowBlur = 20;
        p.drawingContext.shadowColor = 'rgba(255, 220, 0, 0.6)';
      }
    } else if (isHighlighted) {
      p.strokeWeight(3);
      p.stroke(100, 200, 255);
      if (p.drawingContext) {
        p.drawingContext.shadowBlur = 10;
        p.drawingContext.shadowColor = 'rgba(100, 200, 255, 0.4)';
      }
    } else {
      p.strokeWeight(2);
      p.stroke(200, 200, 200);
      if (p.drawingContext) {
        p.drawingContext.shadowBlur = 0;
      }
    }
    
    // Draw rounded rectangle for tube
    p.rect(this.x, drawY, this.width, this.totalHeight, 0, 0, 15, 15);
    
    // Reset shadow
    if (p.drawingContext) {
      p.drawingContext.shadowBlur = 0;
    }
    p.pop();
    
    // Draw water layers
    this.drawWaterLayers(yOffset);
  }
  
  drawWaterLayers(yOffset = 0) {
    const p = this.p;
    const startY = this.y + yOffset;
    
    for (let i = 0; i < this.colors.length; i++) {
      const color = this.colors[i];
      const rgb = COLOR_MAP[color] || [200, 200, 200];
      
      p.push();
      p.noStroke();
      p.fill(...rgb);
      
      const layerY = startY + this.totalHeight - (i + 1) * this.layerHeight;
      
      if (i === 0) {
        // Bottom layer with rounded corners
        p.rect(this.x, layerY, this.width, this.layerHeight, 0, 0, 15, 15);
      } else {
        p.rect(this.x, layerY, this.width, this.layerHeight);
      }
      p.pop();
    }
  }
  
  drawPartialLayer(color, amount) {
    // Draw a partial layer for animation
    const p = this.p;
    const rgb = COLOR_MAP[color] || [200, 200, 200];
    
    p.push();
    p.noStroke();
    p.fill(...rgb);
    
    const layerY = this.y + this.totalHeight - this.colors.length * this.layerHeight - amount * this.layerHeight;
    
    p.rect(this.x, layerY, this.width, amount * this.layerHeight);
    p.pop();
  }
  
  getTopColor() {
    if (this.colors.length === 0) return null;
    return this.colors[this.colors.length - 1];
  }
  
  getTopContiguousLayer() {
    if (this.colors.length === 0) return { color: null, amount: 0 };
    
    const topColor = this.colors[this.colors.length - 1];
    let amount = 0;
    
    for (let i = this.colors.length - 1; i >= 0; i--) {
      if (this.colors[i] === topColor) {
        amount++;
      } else {
        break;
      }
    }
    
    return { color: topColor, amount };
  }
  
  getRemainingCapacity() {
    return this.capacity - this.colors.length;
  }
  
  isEmpty() {
    return this.colors.length === 0;
  }
  
  isFull() {
    return this.colors.length === this.capacity;
  }
  
  isSorted() {
    if (this.colors.length === 0) return true;
    if (this.colors.length !== this.capacity) return false;
    
    const firstColor = this.colors[0];
    return this.colors.every(c => c === firstColor);
  }
  
  canPourInto(destinationTube) {
    if (this.isEmpty()) return false;
    if (destinationTube.isFull()) return false;
    
    const sourceLayer = this.getTopContiguousLayer();
    const destTopColor = destinationTube.getTopColor();
    
    // Check color compatibility
    if (destTopColor !== null && destTopColor !== sourceLayer.color) {
      return false;
    }
    
    // Check capacity
    if (destinationTube.getRemainingCapacity() < sourceLayer.amount) {
      return false;
    }
    
    return true;
  }
  
  pour(destinationTube) {
    if (!this.canPourInto(destinationTube)) return null;
    
    const layer = this.getTopContiguousLayer();
    
    // Remove from source
    for (let i = 0; i < layer.amount; i++) {
      this.colors.pop();
    }
    
    // Add to destination
    for (let i = 0; i < layer.amount; i++) {
      destinationTube.colors.push(layer.color);
    }
    
    return layer;
  }
}

export function canPour(sourceTube, destTube) {
  return sourceTube.canPourInto(destTube);
}