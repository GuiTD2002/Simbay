from typing import Tuple
from typing import Union

import numpy as np

from src.estimation import BaseMeasurementModel
from src.estimation import BaseMotionModel

# Create a custom Type Alias so the code is easy to read
Bounds = float | np.ndarray

class ParticleFilterRegularized:
    def __init__(
        self, 
        num_particles: int, 
        state_bounds: tuple[Bounds, Bounds],
        motion_model: BaseMotionModel, 
        measurement_model: BaseMeasurementModel,
        ess_threshold_ratio: float = 0.5,
        bound_enforcer=None,  
        mean_estimator=None
    ):
        self.N = num_particles
        self.motion_model = motion_model
        self.measurement_model = measurement_model
        self.ess_threshold_ratio = ess_threshold_ratio
        
        # MAGIC TRICK: Convert bounds to 1D arrays immediately.
        # If you pass a float (0.5), it becomes np.array([0.5]). 
        # If you pass an array, it stays an array.
        self.min_bound = np.atleast_1d(state_bounds[0])
        self.max_bound = np.atleast_1d(state_bounds[1])
        self.dim = len(self.min_bound)
        
        # Inline type hinting keeps the top of the class clutter-free
        self.particles: np.ndarray = np.random.uniform(self.min_bound, self.max_bound, size=(self.N, self.dim))
        self.weights: np.ndarray = np.ones(self.N) / self.N

        self.history = {'particles': [], 'estimates': [], 'weights': []}

        self.bound_enforcer = bound_enforcer
        self.mean_estimator = mean_estimator

    def reset(self, state: dict) -> None:
        """Wipes the belief state and scatters particles uniformly."""
        self.particles = np.random.uniform(self.min_bound, self.max_bound, size=(self.N, self.dim))
        self.weights = np.ones(self.N) / self.N
        self.motion_model.change_internal_state(self.particles, state)

    def update_internal_state(self, initial_state: dict) -> None:
        self.motion_model.change_internal_state(self.particles, initial_state)

    def get_history(self) -> dict[str, list[np.ndarray]]:
        return self.history

    
    def estimate(self) -> np.ndarray:
        """Calculates the weighted Expected Value."""
        if self.mean_estimator:
            return self.mean_estimator(self.particles, self.weights)
        
        # Default Generic Euclidean Logic
        return np.average(self.particles, weights=self.weights, axis=0)

    def predict(self, control_input) -> None:
        self.particles = self.motion_model.propagate(self.particles, control_input)
        
    def update(self, observation: dict) -> None:
        """
        Updates particle weights based on the likelihood of the new observation, 
        then normalizes the weights so they sum to 1.
        
        Args:
            observation: The actual sensor reading from the real world or target system.
        """
        likelihoods = self.measurement_model.compute_likelihoods(self.particles, observation)
        likelihoods = np.nan_to_num(likelihoods, nan=0.0, posinf=0.0, neginf=0.0)
        current_weights = np.nan_to_num(self.weights, nan=0.0, posinf=0.0, neginf=0.0)
        if current_weights.sum() <= 0.0:
            current_weights = np.ones(self.N) / self.N

        new_weights = current_weights * likelihoods
        # ==========================================
        # TRUE BULLSEYE DETECTOR
        # ==========================================
        
        if observation.get('contact', 0) == 1:
            
            # 1. Find the highest accumulated weight in the swarm
            max_weight = new_weights.max()
            
            # 2. A True Perfect Particle must hit the bullseye THIS frame (likelihood == 1.0)
            # AND it must not be a Zombie (its accumulated weight must equal the max_weight)
            # (We multiply by 0.99 just to allow for microscopic floating-point rounding errors)
            perfect_mask = (likelihoods >= 0.99) & (new_weights >= max_weight * 0.99) & (max_weight > 0)
            contact_mask = likelihoods >= 0.99
            
            num_perfect = perfect_mask.sum()
            
            if num_perfect > 0:
                print(f"🎯 BULLSEYE: Found {num_perfect} TRUE perfect particle(s) with no past penalties!")
            else:
                # If we have 0 perfect particles, it means the ones that hit were Zombies, 
                # or the ones that survived the sweep missed the contact!
                print(f"📉 No true perfect particles. Best surviving weight: {max_weight:.2e}")
            self._print_contact_debug(observation, likelihoods, new_weights, contact_mask, perfect_mask)
        
        # Commit the new weights
        self.weights = new_weights
        
        # Normalize the weights so they represent a valid probability distribution (summing to 1)
        sum_weights = self.weights.sum()
        if sum_weights == 0.0 or not np.isfinite(sum_weights):
            print("⚠️ CRITICAL WARNING: Particle Extinction Event! All weights collapsed to 0.0. Forcing a uniform reset.")
            self.weights = np.full_like(self.weights, 1.0 / self.N)
        else:
           self.weights /= sum_weights 

    def resample(self, current_state, step=None):
        """
        We use Gaussian Kernel instead of Epanechnikov for efficiency purposes. 
        The Gaussian is almost as good and much faster.
        """
        # Compute the Effective Sample Size (ESS)
        Neff = 1. / np.sum(self.weights**2)

        # Only resample if ESS drops below threshold
        threshold = self.N * self.ess_threshold_ratio
        if Neff < threshold:
            step_msg = f", step={step}" if step is not None else ""
            print(f"[DEBUG] Resampling particles: ESS={Neff:.2f}, threshold={threshold:.2f}{step_msg}")

            # 1. Compute covariance matrix S_k
            nx = self.particles.shape[1]
            S_k = np.cov(self.particles.T, aweights=self.weights, bias=True)

            # Automatically upgrade a 0D scalar to a 1x1 2D matrix
            # (If it is already a 2x2 matrix, this safely does nothing)
            S_k = np.atleast_2d(S_k)

            # (Adding a tiny epsilon to the diagonal prevents LinAlgError if 
            # particles have already collapsed to a single point)
            S_k += np.eye(nx) * 1e-8 

            # 2. Compute D_k such that D_k * D_k.T = S_k
            D = np.linalg.cholesky(S_k)

            # 3. Perform Systematic Resampling
            # Systematic Resampling: Instead of spinning a roulette wheel N times,
            # we spin a wheel with N equally spaced pointers exactly once (offset 'u').
            # This is significantly faster and mathematically more stable.
            u = np.random.rand() 
            positions = (np.arange(self.N) + u) / self.N

            cumulative_sum = np.cumsum(self.weights)
            
            # Guard against floating-point rounding errors that could cause index out-of-bounds
            cumulative_sum[-1] = 1.0  

            indexes = np.searchsorted(cumulative_sum, positions, side='right')
            resampled_particles = self.particles[indexes]
            
            # Reset weights back to uniform for the surviving clones
            self.weights.fill(1.0 / self.N)

            # 4. Calculate the Optimal Bandwidth (h_opt)
            # Using Silverman's rule of thumb for a Gaussian Kernel
            A = (4.0 / (nx + 2.0)) ** (1.0 / (nx + 4.0))
            h_opt = A * (self.N ** (-1.0 / (nx + 4.0)))

            # 5. Generate the Raw Jitter (epsilon)
            # Draw N random vectors from a standard normal distribution
            epsilon = np.random.randn(self.N, nx)

            # 6. Apply the Jitter
            # We transpose epsilon so we can dot-product it with D, 
            # then transpose it back to match the particle array shape.
            # Formula: x* = x + h * D * epsilon
            jitter = h_opt * (D @ epsilon.T).T
            self.particles = resampled_particles + jitter * 1

            # 7. Reset the weights
            # Because we just resampled, all particles now represent equal probability mass
            self.weights = np.ones(self.N) / self.N

            # 8. Clip particles to state bounds
            if self.bound_enforcer:
                self.particles = self.bound_enforcer(self.particles)
            else:
                # Default Generic Euclidean Logic
                self.particles = np.clip(self.particles, self.min_bound, self.max_bound)

            # 8. Update particles in the motion model
            self.motion_model.change_internal_state(self.particles, current_state)

    def _print_contact_debug(self, observation, likelihoods, new_weights, contact_mask, perfect_mask):
        direction = np.asarray(observation.get('direction', []), dtype=float)
        arm_pos = np.asarray(observation.get('arm_pos', []), dtype=float)
        step_size = observation.get('step_size', None)

        print("   [Contact Debug]")
        if direction.size > 0:
            print(f"      sweep direction: {np.array2string(direction, precision=3)}")
        if arm_pos.size > 0:
            print(f"      real arm pos: {np.array2string(arm_pos, precision=4)}")
        if step_size is not None:
            print(f"      step size: {step_size:.6f}")

        self._print_particle_group_debug("likelihood==1 particles", contact_mask, new_weights)
        self._print_particle_group_debug("true-perfect particles", perfect_mask, new_weights)

        top_count = min(10, self.N)
        top_idx = np.argsort(new_weights)[-top_count:][::-1]
        top_particles = self.particles[top_idx]
        top_weights = new_weights[top_idx]
        print(f"      top {top_count} weighted particles:")
        for rank, (particle, weight, likelihood) in enumerate(
            zip(top_particles, top_weights, likelihoods[top_idx]), start=1
        ):
            print(
                f"         #{rank}: p={np.array2string(particle, precision=4)} "
                f"w={weight:.3e} likelihood={likelihood:.3f}"
            )

    def _print_particle_group_debug(self, label, mask, weights):
        count = int(np.sum(mask))
        print(f"      {label}: {count}")
        if count == 0:
            return

        selected = self.particles[mask]
        selected_weights = weights[mask]
        min_vals = selected.min(axis=0)
        max_vals = selected.max(axis=0)
        mean_vals = selected.mean(axis=0)
        spread_vals = max_vals - min_vals
        weight_sum = selected_weights.sum()

        print(f"         min:    {np.array2string(min_vals, precision=4)}")
        print(f"         max:    {np.array2string(max_vals, precision=4)}")
        print(f"         mean:   {np.array2string(mean_vals, precision=4)}")
        print(f"         spread: {np.array2string(spread_vals, precision=4)}")
        print(f"         raw weight sum before normalize: {weight_sum:.3e}")

    def step(self, control_input, observation, current_state):
        self.predict(control_input)
        self.update(observation)
        self.resample(current_state)

    def record_state(self):
        self.history['particles'].append(self.particles.copy())
        self.history['estimates'].append(self.estimate())
        self.history['weights'].append(self.weights.copy())

