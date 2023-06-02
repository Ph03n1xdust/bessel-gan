from typing import Any, Callable, Sequence, Optional
import jax
from ase import Atoms
import numpy as np
from besselgan.models import CriticModel, GeneratorModel


def get_max_R_CUT(atoms_arr: Sequence[Atoms]):
    """Calculates the maximum possible cutoff radius for a list of structures
        with periodic boundary conditions.

    Args:
        atoms_arr: Sequence of atoms objects corresponding to the structures.

    Returns:
        The maximum possible cutoff radius for the specific set of structures.
    """
    result = float("inf")
    for atoms in atoms_arr:
        volume = atoms.get_volume()

        side_areas = []
        for a in atoms.cell[:]:
            for b in atoms.cell[:]:
                side_areas.append(np.sum(np.cross(a, b) ** 2) ** 0.5)
        max_area = np.max(side_areas)
        curr_r_cut = volume / max_area / 2
        result = min(curr_r_cut, result)

    return result


def create_evaluate_single_descriptor(critic: CriticModel):
    """Creates a function to evaluate the descriptors of a single atom.

    Args:
        critic: The CriticModel object used for evaluation.

    Returns:
        A function which evaluates the critic with given weights
        on a specific descriptor.
    """

    @jax.jit
    def evaluate_single_descriptor(params_crit, descriptor):
        return critic.apply(params_crit, descriptor)

    return evaluate_single_descriptor


def create_evaluate_batch_descriptor(critic: CriticModel):
    """Creates a function to evaluate a batch of descriptors.

    Args:
        critic: The CriticModel object used for evaluation.

    Returns:
        A function which evaluates the critic with given weights
        on a batch of descriptor.
    """
    evaluate_single_descriptor = create_evaluate_single_descriptor(critic)

    return jax.jit(jax.vmap(evaluate_single_descriptor, (None, 0), 0))


def create_generate_descriptor(descriptor_method: Callable):
    """Creates a function to generate descriptors of a single structure.

    Args:
        descriptor_method: A function used to generate descriptors.

    Returns:
        A function which generates descriptors of the chosen atoms in a given
        structure. It uses five arguments, the position of all atoms, the types
        of all atoms, the positions of the chosen atoms, the types of the
        chosen atoms and the unit cell respectively.
    """

    @jax.jit
    def generate_descriptor(allpos, alltype, pos, type, cell):
        desc = descriptor_method(allpos, alltype, pos, type, cell)
        return desc.reshape((pos.shape[0], -1))

    return generate_descriptor


def create_generate_batch_descriptor(descriptor_method: Callable):
    """Creates a function to generate descriptors of a batch of structures.

    Args:
        descriptor_method: A function used to generate descriptors.

    Returns:
        A function which generates descriptors of the chosen atoms in a batch
        of structures. It uses five arguments, the position of all atoms, the
        types of all atoms, the positions of the chosen atoms, the types of the
        chosen atoms and the unit cell respectively.
    """

    def generate_descriptor(allpos, alltype, pos, type, cell):
        desc = descriptor_method(allpos, alltype, pos, type, cell)
        return desc.reshape((pos.shape[0], -1))

    return jax.jit(jax.vmap(generate_descriptor))


def create_generate_structures(
    generator: GeneratorModel, postprocess: Callable, n_latent: int
):
    """Creates a function to generate a batch of structures.

    Args:
        generator: The GeneratorModel object used to generate the input of the
            postprocessor.
        postprocess: A function which creates a structure based on the
            generator output.
        n_latent: The number of latent variables.

    Returns:
        A function which generates n_batch structures given the generator
        weights and (n_batch*n_latent) latent variables. The generated
        structures consist of: the positions of all atoms, the types of all
        atoms, the positions of the chosen atoms, the types of the chosen atoms
        and the unit cell respectively.
    """

    def generate_single(params_gen, latent):
        intermediate = generator.apply(params_gen, latent)
        all_pos, all_type, pos, type, cell = postprocess(intermediate)

        return all_pos, all_type, pos, type, cell

    generate_batch = jax.jit(jax.vmap(generate_single, (None, 0), 0))

    def generate_structures(generator_params, key, n_struct):
        latent = jax.random.normal(key, shape=(n_struct, n_latent))
        return generate_batch(generator_params, latent)

    return generate_structures
