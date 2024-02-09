# import asyncio

# from flowdapt_tridium_plugin.api.client import TridiumJACEClient


# # Don't forget to run `export $(cat secrets.env)` or it won't work

# async def main():
#     async with TridiumJACEClient() as client:
#         data, current_time = await client.get_data()

#         print(f"Data as recent as: {current_time}")
#         print(f"Data shape: {data.shape}")
#         print("\nSample:\n")
#         print(data)


# if __name__ == "__main__":
#     asyncio.run(main())
