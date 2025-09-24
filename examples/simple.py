from dotenv import load_dotenv

from openelectricity import OEClient

load_dotenv()


def main():
    client = OEClient()
    results = client.get_facilities()

    for r in results.data:
        # sum up the capacity of the units
        capacity = sum(unit.capacity_registered for unit in r.units if unit.capacity_registered is not None)
        print(f"{r.name} has capacity {capacity}MW")


if __name__ == "__main__":
    main()
