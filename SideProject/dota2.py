from dota2.client import Dota2Client
from steam.client import SteamClient
import logging

logging.basicConfig(format='[%(asctime)s] %(levelname)s %(name)s: %(message)s', level=logging.DEBUG)


client = SteamClient()
dota = Dota2Client(client)

@client.on('logged_on')
def start_dota():
    dota.launch()

@dota.on('ready')
def do_dota_stuff():
    print("!")
    # talk to GC

client.cli_login()
client.run_forever()