import argparse
import os

from device_info import check_device
from env_data_collector import EnvDataCollector
from trainer import RLTrainer
from evaluator import RLEvaluator
from random_renderer import render_random_actions


def main():
    parser = argparse.ArgumentParser(description="RL Project CLI")

    # Define subcommands: collect, train, test, render
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")

    # ---------------------------
    # Sub-parser for data-collect
    # ---------------------------
    collect_parser = subparsers.add_parser("collect", help="Collect environment data in CSV")
    collect_parser.add_argument("--env_id", default="LunarLander-v2", help="Gym environment ID")
    collect_parser.add_argument("--steps", type=int, default=1000, help="Number of steps to collect data")
    collect_parser.add_argument("--csv_filename", default="env_data.csv", help="CSV filename to store data")

    # ---------------------------
    # Sub-parser for training
    # ---------------------------
    train_parser = subparsers.add_parser("train", help="Train an RL model")
    train_parser.add_argument("--env_id", default="LunarLander-v2", help="Gym environment ID")
    train_parser.add_argument("--model_type", default="A2C", choices=["A2C", "PPO"], help="Type of model to train")
    train_parser.add_argument("--timesteps", type=int, default=200_000, help="Number of training timesteps")
    train_parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments")
    train_parser.add_argument("--normalize", action='store_true', help="Whether to use VecNormalize")

    # Make these optional; if not provided, auto-generate from (model_type, env_id)
    train_parser.add_argument("--model_path", default=None, help="Path to save the trained model (optional).")
    train_parser.add_argument("--vecnormalize_path", default=None, help="Path to save VecNormalize stats (optional).")

    # ---------------------------
    # Sub-parser for testing
    # ---------------------------
    test_parser = subparsers.add_parser("test", help="Evaluate a trained RL model")
    test_parser.add_argument("--env_id", default="LunarLander-v2", help="Gym environment ID")
    test_parser.add_argument("--model_type", default="A2C", choices=["A2C", "PPO"], help="Type of model to load")
    test_parser.add_argument("--model_path", default=None, help="Path to the saved model (optional).")
    test_parser.add_argument("--vecnormalize_path", default=None, help="Path to saved VecNormalize stats (optional).")
    test_parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments")
    test_parser.add_argument("--normalize", action='store_true', help="Whether VecNormalize was used during training")
    test_parser.add_argument("--eval_episodes", type=int, default=5, help="Number of evaluation episodes")

    # ---------------------------
    # Sub-parser for rendering
    # ---------------------------
    render_parser = subparsers.add_parser("render", help="Render a trained RL model")
    render_parser.add_argument("--env_id", default="LunarLander-v2", help="Gym environment ID")
    render_parser.add_argument("--model_type", default="A2C", choices=["A2C", "PPO", "null"], 
                               help="Type of model to load (or 'null' to render random actions)")
    render_parser.add_argument("--model_path", default=None, help="Path to the saved model (optional).")
    render_parser.add_argument("--vecnormalize_path", default=None, help="Path to saved VecNormalize stats (optional).")
    render_parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments (1 recommended for rendering)")
    render_parser.add_argument("--normalize", action='store_true', help="Whether VecNormalize was used during training")
    render_parser.add_argument("--steps", type=int, default=1000, help="Number of steps to render")

    #
    # Sub-parser for checking device
    #
    device_parser = subparsers.add_parser("device", help="Check available compute devices")


    args = parser.parse_args()

    # If no subcommand provided, print help
    if args.command is None:
        parser.print_help()
        return

    # ---------------------------
    # Handle each subcommand
    # ---------------------------
    if args.command == "collect":
        collector = EnvDataCollector(env_id=args.env_id, csv_filename=args.csv_filename)
        collector.collect_data(num_steps=args.steps)

    elif args.command == "train":
        # Train a model
        trainer = RLTrainer(
            env_id=args.env_id,
            model_type=args.model_type,
            n_envs=args.n_envs,
            total_timesteps=args.timesteps,
            normalize_env=args.normalize
        )
        trainer.train()
        trainer.save(
            model_filename=args.model_path, 
            vecnormalize_filename=args.vecnormalize_path
        )

    elif args.command == "test":
        # Test/evaluate a model
        trainer = RLTrainer(
            env_id=args.env_id,
            model_type=args.model_type,
            n_envs=args.n_envs,
            total_timesteps=0,  # Not actually training
            normalize_env=args.normalize
        )
        trainer.load(
            model_filename=args.model_path, 
            vecnormalize_filename=args.vecnormalize_path
        )

        if trainer.model is None:
            print("[Main] No model found to test. Exiting.")
            return

        evaluator = RLEvaluator(trainer.model)
        evaluator.evaluate_model(n_eval_episodes=args.eval_episodes)

    elif args.command == "device":
        check_device()
        return

    elif args.command == "render":
        # If the user passes "null" for model_type, do not load a model, just render random actions
        if args.model_type.lower() == "null":
            render_random_actions(
                env_id=args.env_id,
                n_steps=args.steps,
                n_envs=args.n_envs,
                normalize=args.normalize,
                vecnormalize_path=args.vecnormalize_path
            )
            return

        # Otherwise, load and render a model
        trainer = RLTrainer(
            env_id=args.env_id,
            model_type=args.model_type,
            n_envs=args.n_envs,
            total_timesteps=0,
            normalize_env=args.normalize
        )
        trainer.load(
            model_filename=args.model_path, 
            vecnormalize_filename=args.vecnormalize_path
        )

        if trainer.model is None:
            print("[Main] No model found to render. Exiting.")
            return

        evaluator = RLEvaluator(trainer.model)
        evaluator.render_model(n_steps=args.steps, env_id= args.env_id)


if __name__ == "__main__":
    main()
