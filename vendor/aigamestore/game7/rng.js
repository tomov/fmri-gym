// Seeded Random Number Generator for deterministic replay
// Uses a simple Linear Congruential Generator (LCG) algorithm

let seed = 42;
let state = seed;

/**
 * Set the seed for the RNG
 * @param {number} newSeed - The seed value
 */
export function setSeed(newSeed) {
  seed = newSeed;
  state = newSeed;
}

/**
 * Get the current seed
 * @returns {number} The current seed
 */
export function getSeed() {
  return seed;
}

/**
 * Generate a random number between 0 and 1 (exclusive of 1)
 * Uses LCG: state = (state * 1664525 + 1013904223) % 2^32
 * @returns {number} Random number between 0 and 1
 */
export function random() {
  // LCG parameters (from Numerical Recipes)
  state = (state * 1664525 + 1013904223) >>> 0; // Use unsigned right shift to ensure 32-bit
  // Convert to float between 0 and 1
  return (state >>> 0) / 0xFFFFFFFF;
}

/**
 * Generate a random number between min and max (exclusive of max)
 * @param {number} min - Minimum value (inclusive)
 * @param {number} max - Maximum value (exclusive)
 * @returns {number} Random number between min and max
 */
export function randomRange(min, max) {
  if (max === undefined) {
    max = min;
    min = 0;
  }
  return min + random() * (max - min);
}

/**
 * Generate a random integer between min and max (inclusive)
 * @param {number} min - Minimum value (inclusive)
 * @param {number} max - Maximum value (inclusive)
 * @returns {number} Random integer between min and max
 */
export function randomInt(min, max) {
  if (max === undefined) {
    max = min;
    min = 0;
  }
  return Math.floor(random() * (max - min + 1)) + min;
}

/**
 * Reset the RNG state to the current seed
 * Useful for ensuring determinism after loading a game state
 */
export function reset() {
  state = seed;
}

// Initialize with default seed
setSeed(42);
