import abilities
import systems
import vernon_dsl as vd

asset = vd.program_asset(id="demo/systems", program=(systems, abilities), prefix="demo_")
