local host, port = "127.0.0.1", 8001
local socket = require("socket")
local tcp = assert(socket.tcp())
tcp:connect(host, port)
tcp:send('Connected')

local BUTTONS = {0, 0, 0, 0, 0, 0, 0, 0, 0}
local OUT_TO_BUTTONS = {
    {0, 0, 0, 0, 0, 1, 0, 0, 0},
    {1, 0, 0, 0, 0, 1, 0, 0, 0},
    {0, 1, 0, 0, 0, 1, 0, 0, 0}
}
local tracks = {
	"Crash Cove",
	"Roo's Tubes",
	"Tiger Temple",
	"Coco Park",
	"Mystery Caves",
	"Blizzard Bluff",
	"Sewer Speedway",
	"Dingo Canyon",
	"Papu's Pyramid",
	"Dragon Mines",
	"Polar Pass",
	"Cortex Castle",
	"Tiny Arena",
	"Hot Air Skyway",
	"N.gin Labs",
	"Oxide Station",
	"Slide Coliseum",
	"Turbo Track"
}
local COEFFX = {{1096.3871668881495, 0.00024703},
		{648.8907972725096, 0.00016692},
		{808.2214949463429, 0.00015567},
		{1067.4476422000566, 0.00022205},
		{553.2231225619722, 0.00010774},
		{679.2222767804917, 0.00016676},
		{462.3120376807265, 0.00012186},
		{249.2599706607075, 0.00020405},
		{714.8400101850771, 0.00019952},
		{612.209940534515, 0.00022046},
		{518.8995554270739, 0.00015246},
		{661.4728181950248, 0.00016822},
		{581.7717707204861, 0.00013887},
		{587.3032686666095, 0.00019695},
		{258.24418337102253, 0.00017157},
		{317.87574198039175, 0.0001749},
		{676.2005498317476, 0.00011718},
		{806.3893287263951, 0.00014062}
		}
local COEFFY = {{720.6058956589455, 0.00024687},
		{951.6204588111457, 0.00016692},
		{541.2663631660163, 0.00015532},
		{622.7371313353979, 0.00022273},
		{528.5968185620499, 0.00010798},
		{383.9597198513739, 0.00016832},
		{978.7461432835855, 0.0001207},
		{495.31945787523387, 0.00020356},
		{700.0342431302049, 0.00019857},
		{730.9846413033176, 0.00021962},
		{1128.8737994356275, 0.00015239},
		{560.0389412399034, 0.00016791},
		{491.5447563291968, 0.000139},
		{703.7373943101733, 0.0001968},
		{9.809855177047098, 0.00017184},
		{308.57698624296677, 0.00017524},
		{546.7318019831731, 0.00011678},
		{542.7879343976957, 0.00013991}
		}
local SAVESTATESLOT = 0

local function press_buttons(frame, raceended, tnt, position, ram_spd, jump_switch, turbo_flag, turbo_charge, weapon)
	circle = 0
	l1 = 0
	r1 = 0
	down = 0
	r2 = 0
	-- use item every 3 seconds or when item is rolling to get it faster
	if weapon > 0 then
		if weapon < 65 then
			circle = 1
		end
	elseif (raceended == 0) and (frame < 10) then
		circle = 1
	end
	-- if first, look backwards and shoot item backwards (back bomb and missile)
	if (raceended == 0) and (position == 0) and ((frame > 170) or (frame < 30)) then
		down = 1
		r2 = 1
	end
	-- froggy if TNT (or speed high enough)
	if (raceended == 0) and ((tnt > 0) or ((ram_spd > 15000) and (turbo_flag ~=2))) then
		if (frame%4 < 2) then
			r1 = 1
		else
			l1 = 1
		end
	-- jump off edges
	elseif (raceended == 0) and (turbo_charge == 0) and (jump_switch == 5) and (ram_spd > 10000) and (turbo_flag ~=2) then
		r1 = 1
	end
	joypad.set({Left=BUTTONS[1], Right=BUTTONS[2], L1=l1, R1=r1, R2=r2, Square=BUTTONS[5], Cross=BUTTONS[6], Up=BUTTONS[7], Down=down, Start=BUTTONS[9], Circle=circle}, 1)

end

local function has_value(tab, val)
	for index, value in ipairs(tab) do
		if value == val then
			return true
		end
	end
	return false
end


function average (a)
	local sum = 0
	local count = 0.0
	for i,v in ipairs(a) do
		if v ~= 0 then
			sum = sum + v
			count = count + 1
		end
	end
	return sum / count
end

function joypad_y(oldy)
	return newy1 + ((oldy-oldy1)/(oldy2-oldy1))*(newy2-newy1)
end

function joypad_x(oldx)
	return newx1 + ((oldx-oldx1)/(oldx2-oldx1))*(newx2-newx1)
end

function Bit(num, n_bit)
    -- returns bit in position n_bit
    local t={} -- will contain the bits
    while num>0 do
        rest=math.fmod(num,2)
        t[#t+1]=rest
        num=(num-rest)/2
    end
    if t[n_bit] == nil then return 0 else return t[n_bit] end
end

local function end_race(frame)
	if (frame % 120 == 60) then BUTTONS = {0, 0, 0, 0, 0, 1, 0, 0, 0}
	elseif (frame % 120 == 61) then BUTTONS = {0, 0, 0, 0, 0, 1, 0, 0, 0}
	else BUTTONS = {0, 0, 0, 0, 0, 0, 0, 0, 0}
	end
end

local function restart_race(frame)
	if frame < 3 then BUTTONS = {0, 0, 0, 0, 0, 0, 0, 0, 1}
	elseif frame < 20 then BUTTONS = {0, 0, 0, 0, 0, 0, 0, 0, 0}
	elseif frame < 23 then BUTTONS = {0, 0, 0, 0, 0, 0, 0, 1, 0}
    elseif frame < 40 then BUTTONS = {0, 0, 0, 0, 0, 0, 0, 0, 0}
	elseif frame < 43 then BUTTONS = {0, 0, 0, 0, 0, 1, 0, 0, 0}
	else BUTTONS = {0, 0, 0, 0, 0, 0, 0, 0, 0}
	end
end

local function load_savestate(SAVESTATESLOT)
	savestate.loadslot(SAVESTATESLOT)
end

local function get_x(x, track_id)
	return x*COEFFX[track_id+1][2] + COEFFX[track_id+1][1]
end

local function get_y(y, track_id)
	return y*COEFFY[track_id+1][2] + COEFFY[track_id+1][1]
end

local X_POS, Y_POS, Z_POS, X_SPD, Y_SPD, Z_SPD
local X_POS_2, Y_POS_2
local JUMP, ANGLE, RNG
local FRAME = 0

local XTEXT = 2
local YTEXT = 60
local XTEXT2 = 600

local XBOX = 670
local YBOX = 55

local XBOX2 = 10
local YBOX2 = 350

local PREV_LAPPROG, LAPPROG, LAPCOUNTER, RACEENDED, TIMER, DRIVEBACKWARDS, ISBACKWARDS
local PICKUP, PREV_PICKUP, WALL
local FRAMESENDED = 0
local ENDSWITCH = 1
local FRAME_SKIP_COUNTER = 0
local rewards = {}
local finish = {}
local total_reward = 0.
local best_reward = 0
local n_episode = 0
local finished_attempts = 0
local finished_attempts_track = {}
for i = 1,18,1
do
	finished_attempts_track[i] = 0
end
local average_track = {}
for i = 1,36,1
do
	average_track[i] = {}
end
local best_track = {}
for i = 1,36,1
do
	best_track[i] = 9999999
end
local average_finish_track = {}
for i = 1,18,1
do
	average_finish_track[i] = 0
end
local average_time_track = {}
for i = 1,18,1
do
	average_time_track[i] = 0
end
local best_finish_track = {}
for i = 1,18,1
do
	best_finish_track[i] = 9999999
end
local best_time_track = {}
for i = 1,18,1
do
	best_time_track[i] = 9999999
end
local average_size = 64
local average_size_track = 6
local average_reward = 0
local average_finish = 0.
local is_random = 0
local random_rate = 0.
local reward_malus = 0.
local TRACK = 0.
local IMAGE_PATH
local FRAMES_NOT_MOVING = 0
local TIMER_LIMIT = 0
local TURBO_FLAG = 0
local POSITION = 0
local TNT = 0
local JUMP_SWITCH = 0
local WEAPON = 0
oldx1 = 146
oldx2 = 346
oldy1 = 314
oldy2 = 456
newx1 = 2
newx2 = 152
newy1 = 340
newy2 = 460

-- modes: Cup, Race, Adventure
local MODE = 'Cup'

local HASH = gameinfo.getromhash()
print(HASH)



while true do
	FRAME = (FRAME + 1) % 180
	--memory.write_s8(0x8D2B4, 3)
	--POINTER = memory.read_u32_le( 0x08D674 ) - 0x80000000
	--POINTER = memory.read_u32_le( 0x097FFC ) - 0x80000000
	if (HASH=='68A605C9') then POINTER = memory.read_u32_le( 0x09900C ) - 0x80000000 end		--US Version
	if (HASH=='5C0D56EF') then POINTER = memory.read_u32_le( 0x0993CC ) - 0x80000000 + 0x4  end	--PAL Version
	if (HASH=='07FE354E') then POINTER = memory.read_u32_le( 0x09C4CC ) - 0x80000000 + 0x4  end	--JAP Version
	--POINTER = memory.read_u32_le( 0x09902C ) - 0x80000000
	--POINTER = memory.read_u32_le( 0x1FFE30 ) - 0x80000000


	if (POINTER > 0) then

		X_POS_2 = X_POS
		Y_POS_2 = Y_POS

		PREV_PICKUP = PICKUP

		X_SPD = 		(memory.read_s16_le( POINTER + 0x3A0 ))	-- + 0x88
		Y_SPD = 		(memory.read_s16_le( POINTER + 0x3A8 ))	-- + 0x90
		Z_SPD = 		(memory.read_s16_le( POINTER + 0x3A4 ))	-- + 0x8C
		X_POS = 		(memory.read_s32_le( POINTER + 0x2D4 )) -- + 0x2E0
		Y_POS = 		(memory.read_s32_le( POINTER + 0x2DC )) -- + 0x2E8
		Z_POS = 		(memory.read_s32_le( POINTER + 0x2D8 ))
		RAM_SPD = 		(memory.read_s16_le( POINTER + 0x38C ))
		ANGLE = 		(memory.read_s16_le( POINTER + 0x39A ))	-- + 0x2EE
		TURBO_CHARGE = 	(memory.read_s16_le( POINTER + 0x3DC ))
		TURBO = 		(memory.read_s16_le( POINTER + 0x3E2 ))
		JUMP = 			(memory.read_s8( POINTER + 0x40D ))
		RNG = 			(memory.read_u16_le( 0x8D424 ))
		LAPPROG = 		(memory.read_s32_le( POINTER + 0x488 ))
		if (HASH=='07FE354E') then
			LAPCOUNTER = 		(memory.read_u8( POINTER + 0x1D3F44 - 1916676 ))	
		else
			LAPCOUNTER = 		(memory.read_u8( POINTER + 0x44 ))
		end
		RACEENDED = 		(memory.read_u8( POINTER + 0x2CB )) % 16
		TIMER = 		(memory.read_s32_le( POINTER + 0x514 ))
		DRIVEBACKWARDS =(memory.read_s16_le( POINTER + 0x490 ))
		ISBACKWARDS = Bit((memory.read_s32_le( POINTER + 0x2C8 )), 9)
		if (HASH=='07FE354E') then
			WALL = 			(memory.read_u16_le( POINTER + 0x1D3F56 - 1916676 ))
		else
			WALL = 			(memory.read_u16_le( POINTER + 0x50 ))
		end
		PICKUP = 		(memory.read_s8( POINTER + 0x376  ))
		TURBO_FLAG = (memory.read_u16_le( POINTER + 0xBC  ))
		POSITION = (memory.read_u16_le(POINTER + 0x482))
		TNT = (memory.read_u8(POINTER + 0x1F061C - 2033160))
		WEAPON = (memory.read_u8(POINTER + 0x36))
		if (JUMP < 0) and (JUMP_SWITCH == 0) then
			JUMP_SWITCH = 5
		elseif (JUMP == 0) and JUMP_SWITCH > 0 then
			JUMP_SWITCH = JUMP_SWITCH - 1
		end

		if (Y_POS == Y_POS_2) and (X_POS == X_POS_2) then
			FRAMES_NOT_MOVING = FRAMES_NOT_MOVING + 1
		else
			FRAMES_NOT_MOVING = 0
		end

		if LAPCOUNTER == 0 then
			TIMER_LIMIT = 200000
		elseif LAPCOUNTER == 1 then
			TIMER_LIMIT = 400000
		else
			TIMER_LIMIT = 600000
		end

		-- find which track is played
		if has_value({1772068,1771140,1769548,1767356,1765260,1767580,1771508,1762612,1770696,1768044,1764212,1765368,1808472,1767336,1760880,1768752,1841684}, POINTER) then 
			TRACK = 7 
			IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/dingocanyon_processed.png"
		end--canyon
		if has_value({1927944,1927016,1925424,1923232,1921136,1923456,1927384,1918488,1926572,1923920,1920088,1921244,1964348,1923212,1916756,1924628,2000036}, POINTER) then 
			TRACK = 9 
			if LAPPROG > 26000 then
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/dragonmines_processed.png"
			else 
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/dragonmines_processed2.png"
			end
		end--mines
		if has_value({1945152,1944224,1942632,1940440,1938344,1940664,1944592,1935696,1943780,1941128,1937296,1938452,1981556,1940420,1933964,1941836,2014272}, POINTER) then 
			TRACK = 5 
			IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/blizzardbluff_processed.png"
		end--bluff
		if has_value({1962880,1961952,1960360,1958168,1956072,1958392,1962320,1953424,1961508,1958856,1955024,1956180,1999284,1958148,1951692,1959564,2033160}, POINTER) then 
			TRACK = 0 
			IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/crashcove_processed.png"
		end--cove
		if has_value({1980180,1979252,1977660,1975468,1973372,1975692,1979620,1970724,1978808,1976156,1972324,1973480,2016584,1975448,1968992,1976864,2042868}, POINTER) then 
			TRACK = 2 
			IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/tigertemple_processed.png"
		end--tiger
		if has_value({1954028,1953100,1951508,1949316,1947220,1949540,1953468,1944572,1952656,1950004,1946172,1947328,1990432,1949296,1942840,1950712,2014068}, POINTER) then 
			TRACK = 8 
			if LAPPROG < 53000 then
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/papupyramid_processed.png"
			elseif LAPPROG < 59000 then
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/papupyramid_processed3.png"
			else 
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/papupyramid_processed2.png"
			end
		end--papu
		if has_value({1844080,1843152,1841560,1839368,1837272,1839592,1843520,1834624,1842708,1840056,1836224,1837380,1880484,1839348,1832892,1840764,1916676}, POINTER) then  
			TRACK = 1 
			if LAPPROG > 61000 or LAPPROG < 44000 then
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/rootubes_processed.png"
			else 
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/rootubes_processed2.png"
			end
		end--roo
		if has_value({1959164,1964380,1962788,1960596,1958500,1960820,1964748,1955852,1963936,1961284,1957452,1958608,2001712,1960576,1954120,1961992,2039148}, POINTER) then 
			TRACK = 13 
			if LAPPROG < 63000 and LAPPROG > 43000 then
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/has_processed3.png"
			elseif (LAPPROG > 62900 and LAPPROG < 100800) or (LAPPROG < 26000 and LAPPROG > 15000) then
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/has_processed2.png"
			else
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/has_processed.png"
			end
		end--has
		if has_value({1893092,1892164,1890572,1888380,1886284,1888604,1892532,1883636,1891720,1889068,1885236,1886392,1929496,1888360,1881904,1889776,1953400}, POINTER) then 
			TRACK = 6 
			IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/sewerspeedway_processed.png"
		end--sewer
		if has_value({1967000,1976312,1974720,1972528,1970432,1972752,1976680,1967784,1975868,1973216,1969384,1970540,2013644,1972508,1966052,1973924,2039528}, POINTER) then 
			TRACK = 4 
			IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/mysterycaves_processed.png"
		end--caves
		if has_value({1978560,1977632,1976040,1973848,1971752,1974072,1978000,1969104,1977188,1974536,1970704,1971860,2014964,1973828,1967372,1975244,2047468}, POINTER) then 
			TRACK = 11 
			if LAPPROG < 85000 and LAPPROG > 60000 then
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/cortexcastle_processed.png"
			else 
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/cortexcastle_processed2.png"
			end
		end--castle
		if has_value({1980648,1979720,1978128,1975936,1973840,1976160,1980088,1971192,1979276,1976624,1972792,1973948,2017052,1975916,1969460,1977332,2045100}, POINTER) then 
			TRACK = 14 
			if LAPPROG < 89000 and LAPPROG > 69000 then
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/nginlabs_processed.png"
			else 
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/nginlabs_processed2.png"
			end
		end--labs
		if has_value({1970208,1975424,1973832,1971640,1969544,1971864,1975792,1966896,1974980,1972328,1968496,1969652,2012756,1971620,1965164,1973036,2042736}, POINTER) then 
			TRACK = 10 
			IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/polarpass_processed.png"
		end--pass
		if has_value({1977872,1976944,1975352,1973160,1971064,1973384,1977312,1968416,1976500,1973848,1970016,1971172,2014276,1973140,1966684,1974556,2045420}, POINTER) then  
			TRACK = 15 
			if LAPPROG < 43000 or LAPPROG > 135000 then
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/oxidestation_processed.png"
			elseif LAPPROG > 95000 then
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/oxidestation_processed2.png"
			elseif LAPPROG > 66000 then
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/oxidestation_processed3.png"
			else
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/oxidestation_processed4.png"
			end
		end--station
		if has_value({1765308,1764380,1762788,1760596,1758500,1760820,1764748,1755852,1763936,1761284,1757452,1758608,1801712,1760576,1754120,1761992,1831380}, POINTER) then  
			TRACK = 3 
			IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/cocopark_processed.png"
		end--park
		if has_value({1977048,1976120,1974528,1972336,1970240,1972560,1976488,1967592,1975676,1973024,1969192,1970348,2013452,1972316,1965860,1973732,2029920}, POINTER) then  
			TRACK = 12 
			if LAPPROG < 38000 or LAPPROG > 137000 then
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/tinyarena_processed.png"
			elseif LAPPROG > 83000 then
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/tinyarena_processed3.png"
			else 
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/tinyarena_processed2.png"
			end
		end--arena
		if has_value({1909748,1910232,1908640,1906448,1904352,1906672,1910600,1901704,1909788,1907136,1903304,1904460,1947564,1906428,1899972,1907844,1966756}, POINTER) then  
			TRACK = 16 
			if LAPPROG < 84000 and LAPPROG > 40000 then
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/slidecoliseum_processed3.png"
			elseif LAPPROG < 40001 and LAPPROG > 21000 then
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/slidecoliseum_processed2.png"
			else
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/slidecoliseum_processed.png"
			end
		end--slide
		if has_value({1989036,1987444,1985252,1983156,1985476,1989404,1980508,1988592,1985940,1982108,1983264,2026368,1985232,1978776,1986648,1988552}, POINTER) then 
			TRACK = 17 
			if LAPPROG < 41000 and LAPPROG > 36500 then
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/turbotrack_processed2.png"
			elseif LAPPROG < 36501 and LAPPROG > 32000 then
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/turbotrack_processed.png"
			else
				IMAGE_PATH = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/turbotrack_processed3.png"
			end
		end--turbo

		if FRAME_SKIP_COUNTER == 0 then
		 PREV_LAPPROG = LAPPROG
		end
		if WALL > 0 then
			reward_malus = reward_malus + 0.1
		end
		if PICKUP > 3 then
			reward_malus = reward_malus + 50.
		end
		if FRAMES_NOT_MOVING > 0 then
			reward_malus = reward_malus + 0.1
		end

		
		TOT_SPD = math.floor(math.sqrt(X_SPD*X_SPD+Y_SPD*Y_SPD))

		-- gui.text(XTEXT,60,"Angle : " .. ANGLE,"white")
		--gui.text(XTEXT,80,"Speed (RAM) : " .. RAM_SPD,"white")
		--gui.text(XTEXT,100,"Speed (True): " .. TOT_SPD,"white")
		--gui.text(XTEXT,120,"Reserve : " .. TURBO,"white")
		--gui.text(XTEXT,140,"Charge : " .. TURBO_CHARGE,"white")
		--gui.text(XTEXT,160,"Jump : " .. JUMP,"white")
		
		-- gui.text(XTEXT,200,"Charge : " .. string.format("%08X", POINTER + 0x3E2),"white")
		--gui.text(XTEXT,200,string.format("%08X", POINTER),"white")
		
		gui.text(XTEXT,20,"X : " .. X_POS,"white")
		gui.text(XTEXT,40,"Y : " .. Y_POS,"white")
		gui.text(XTEXT,60,"Speed (RAM) : " .. RAM_SPD,"white")
		--gui.text(XTEXT,120,"Turbo Flag: " .. TURBO_FLAG,"white")
		--gui.text(XTEXT,140,"Jump persistence: " .. JUMP_PERSISTENCE,"white")
		gui.text(XTEXT,80,"Time left to complete lap:", "white")
		gui.text(XTEXT,100,TIMER .. "/" .. TIMER_LIMIT, "white")
		--gui.text(XTEXT,280,"Z : " .. Z_POS,"white")
		--gui.text(XTEXT,300,"Angle : " .. ANGLE,"white")
		--gui.text(XTEXT,300,"Track : " .. POINTER,"white")
		gui.text(XTEXT,120,"Lap Progress : " .. LAPPROG,"white")
		--gui.text(XTEXT,120,"Lap " .. LAPCOUNTER+1,"white")
		--gui.text(XTEXT,340,"Drive Backwards : " .. DRIVEBACKWARDS,"white")
		-- if TOT_SPD>20000 then 
			-- savestate.saveslot(9)
			-- print(TOT_SPD) 
		-- end
		
		--gui.drawBox(XBOX, YBOX-30, XBOX + 50, YBOX + 400, "white")
		--gui.drawText(XBOX+60,(YBOX-7),"USF","white")
		--gui.drawText(XBOX+60,(YBOX-7)+400-17996/64,"SF","white")
		--gui.drawText(XBOX+60,(YBOX-7)+400-16972/64,"Turbo 3","white")
		-- gui.drawText(XBOX+56,(YBOX-7)+400-16460/64,"Turbo 2","white")
		--gui.drawText(XBOX+60,(YBOX-7)+400-15948/64,"Turbo 1","white")
		
		--gui.drawLine(XBOX + 50, YBOX, XBOX + 58, YBOX, "white")
		--gui.drawLine(XBOX + 50, YBOX+400-17996/64, XBOX + 58, YBOX+400-17996/64, "white")
		--gui.drawLine(XBOX + 50, YBOX+400-16972/64, XBOX + 58, YBOX+400-16972/64, "white")
		--gui.drawLine(XBOX + 50, YBOX+400-15948/64, XBOX + 58, YBOX+400-15948/64, "white")
		
		--gui.drawLine(XBOX, YBOX+400-TOT_SPD/64, XBOX + 50, YBOX+400-TOT_SPD/64, "red")
		
		--if (math.sqrt((X_POS-X_ANC)^2+(Y_POS-Y_ANC)^2) ~= 0) then
		--	SPD_CALC = math.floor(math.sqrt((X_POS-X_ANC)^2+(Y_POS-Y_ANC)^2))
		--end
		
		-- gui.text(XTEXT,320,"Speed : " .. SPD_CALC,"white")
		-- gui.text(XTEXT,340,"RNG : " .. string.format("%04X", RNG),"white")
		
		--local RESERVE = TURBO/32768*120
		--local COL = "OrangeRed"
		--local DUMMY = 0
		--if RESERVE <0 then 
		--	RESERVE = 120 
		--	COL = "firebrick"
		--	DUMMY = 1
		--end
		
		
		--gui.drawText(XBOX2+34, YBOX2-18, "Reserve")
		--gui.drawText(XBOX2+54-3.5*math.floor(math.log10(math.abs(TURBO+0.1))+DUMMY), YBOX2+22, TURBO)
		--gui.drawBox(XBOX2, YBOX2, XBOX2 + RESERVE, YBOX2 + 20, "black", COL)
		--gui.drawBox(XBOX2, YBOX2, XBOX2 + 120, YBOX2 + 20, "white")
		
		--X_ANC = X_POS
		--Y_ANC = Y_POS
		--if LAPCOUNTER == 1 then
		--	memory.write_s8( POINTER + 0x2CB, 2)
		--end


		table=joypad.getimmediate()
		action = 0
		if is_random == 0 then color = "cyan" else color = "darkred" end
		if table["P1 Left"] then 
			gui.drawBox(joypad_x(164),joypad_y(375),joypad_x(176),joypad_y(385),color,color)
			action = 1 
		else end
		if table["P1 Right"] then 
			gui.drawBox(joypad_x(188),joypad_y(375),joypad_x(200),joypad_y(385),color,color)
			action = 2
		else end
		if table["P1 Up"] then gui.drawBox(joypad_x(177),joypad_y(362),joypad_x(187),joypad_y(377),color,color) else end
		if table["P1 Down"] then gui.drawBox(joypad_x(177),joypad_y(383),joypad_x(187),joypad_y(398),color,color) else end
		if table["P1 Select"] then gui.drawBox(joypad_x(214),joypad_y(377),joypad_x(231),joypad_y(383),color,color) else end
		if table["P1 Start"] then gui.drawBox(joypad_x(261),joypad_y(377),joypad_x(274),joypad_y(383),color,color) else end
		if table["P1 Square"] then gui.drawBox(joypad_x(242),joypad_y(373),joypad_x(298),joypad_y(390),color,color) else end
		if table["P1 Circle"] then gui.drawBox(joypad_x(320),joypad_y(373),joypad_x(334),joypad_y(387),color,color) else end
		if table["P1 Triangle"] then gui.drawBox(joypad_x(302),joypad_y(356),joypad_x(316),joypad_y(370),color,color) else end
		if table["P1 Cross"] then gui.drawBox(joypad_x(300),joypad_y(391),joypad_x(316),joypad_y(406),color,color) else end
		if table["P1 L1"] then gui.drawBox(joypad_x(174),joypad_y(334),joypad_x(190),joypad_y(343),color,color) else end
		if table["P1 L2"] then gui.drawBox(joypad_x(173),joypad_y(314),joypad_x(191),joypad_y(327),color,color) else end
		if table["P1 R1"] then gui.drawBox(joypad_x(301),joypad_y(334),joypad_x(315),joypad_y(343),color,color) else end
		if table["P1 R2"] then gui.drawBox(joypad_x(300),joypad_y(315),joypad_x(320),joypad_y(329),color,color) else end
		gui.drawImage("C:/Users/Justin/Documents/CTR/BizHawk-2.4.1/Test2.png",newx1,newy1,newx2-newx1,newy2-newy1)
		scale = 1.
		gui.drawImageRegion(IMAGE_PATH,(get_x(X_POS, TRACK)-100.)/scale,(get_y(Y_POS, TRACK)-100.)/scale,200/scale,200/scale, 660, 360, 100, 100)
		
		--gui.drawImageRegion("C:/Users/Justin/Documents/CTR/BizHawk-2.4.1/crashcove_aiview.png",get_x(X_POS)-100,get_y(Y_POS)-100,200,200, 2, 350, 100, 100)
		gui.drawPie(660,360,100,100,360*(-ANGLE)/4095 +45 ,90,"yellow", "null")

		if is_random == 2 and ENDSWITCH == 0 and PREV_PICKUP < 4 then --manual training
			tcp:send(action)
		end

		--counter: instead of running the ml model every frame, run it every n frames (speeds up the process)
		FRAME_SKIP_COUNTER = FRAME_SKIP_COUNTER + 1

		if (RACEENDED ~= 2) and (TIMER > 0) and (TIMER < TIMER_LIMIT) and (FRAMES_NOT_MOVING < 1000) and (PREV_PICKUP < 4) and (FRAME_SKIP_COUNTER > 9) then
			if ENDSWITCH == 1 then
				ENDSWITCH = 0
				FRAMESENDED = 0
				total_reward = 0
				n_episode = n_episode + 1
			end
			FRAME_SKIP_COUNTER = 0
			FRAMESENDED = FRAMESENDED + 1
			reward = (PREV_LAPPROG - LAPPROG) / 33.0 * RAM_SPD / 12000.0
			if math.abs(PREV_LAPPROG - LAPPROG) > 5000 then reward = 0 
			end
			if reward_malus >= 50. then
				reward = -50.
			else
				reward = reward - reward_malus
			end
			reward_malus = 0.

			total_reward = total_reward + reward
			
			if FRAMESENDED > 1 then
				tcp:send(reward)
				feedback, status, partial = tcp:receive()
			end
			send_str = ""
			send_str = X_POS .. " " .. Y_POS .. " " .. ANGLE .. " " .. LAPPROG .. " " .. TRACK
	        tcp:send(send_str)
			action, status, partial= tcp:receive()
			if action == nil then
				action = 0
			end
			is_random_str, status, partial = tcp:receive()
			is_random = tonumber(is_random_str)
		    BUTTONS = OUT_TO_BUTTONS[tonumber(action) + 1]
		    random_rate, status, partial= tcp:receive()
		    if random_rate == nil then
				random_rate = 0
			end
		    
		    if is_random ~= 2 then
		    	press_buttons(FRAME, RACEENDED, TNT, POSITION, RAM_SPD, JUMP_SWITCH, TURBO_FLAG, TURBO_CHARGE, WEAPON)
		    else 
		    	joypad.set({Cross = 1}, 1)
		    end
		elseif (RACEENDED == 2) then
			if ENDSWITCH == 0 then
				ENDSWITCH = 1
				FRAMESENDED = 0
				FRAME_SKIP_COUNTER = 10
				tcp:send(5000) --signal to end the episode
				feedback, status, partial = tcp:receive()
				finished_attempts_track[TRACK + 1] = finished_attempts_track[TRACK + 1] + 1
				finish[n_episode % average_size] = POSITION + 1
				average_finish = average(finish)
				average_track[TRACK+1][finished_attempts_track[TRACK+1] % average_size_track] = POSITION + 1
				average_track[TRACK+19][finished_attempts_track[TRACK+1] % average_size_track] = TIMER
				average_finish_track[TRACK+1] = average(average_track[TRACK+1])
				average_time_track[TRACK+1] = average(average_track[TRACK+19])
				if POSITION + 1 < best_finish_track[TRACK+1] then
					best_finish_track[TRACK+1] = POSITION + 1
				end
				if TIMER < best_time_track[TRACK+1] then
					best_time_track[TRACK+1] = TIMER
				end
			end
			FRAMESENDED = FRAMESENDED + 1
			if (MODE == 'Race') or 
				(
					(MODE == 'Cup')
					and
					(FRAMESENDED > 600)
					and
					((TRACK == 3) or (TRACK == 6) or (TRACK == 12) or (TRACK == 16))
				)
			then 
				if (MODE == 'Race') then
					SAVESTATESLOT = (SAVESTATESLOT%9)+1
				elseif (MODE == 'Cup') then
					SAVESTATESLOT = (SAVESTATESLOT%4)+1
				end
				load_savestate(SAVESTATESLOT)
			else
				end_race(FRAMESENDED)
				press_buttons(0, 15, 0, -1, 0, 0, 0, 0, 0)
			end
		elseif (RACEENDED == 0) and ((TIMER >= TIMER_LIMIT) or (FRAMES_NOT_MOVING >= 1000)) then --if too long without progress restart
			if ENDSWITCH == 0 then
				ENDSWITCH = 1
				FRAMESENDED = 0
				FRAME_SKIP_COUNTER = 10
				tcp:send(5000) --signal to end the episode
				feedback, status, partial = tcp:receive()
			end
			FRAMES_NOT_MOVING = 0
			FRAMESENDED = FRAMESENDED + 1
			if (MODE == 'Race') then
				SAVESTATESLOT = (SAVESTATESLOT%9)+1
			elseif (MODE == 'Cup') then
				SAVESTATESLOT = (SAVESTATESLOT%4)+1
			end
			load_savestate(SAVESTATESLOT)
		else --keep pressing same button during skip frames
			BUTTONS = OUT_TO_BUTTONS[tonumber(action) + 1]
			press_buttons(FRAME, RACEENDED, TNT, POSITION, RAM_SPD, JUMP_SWITCH, TURBO_FLAG, TURBO_CHARGE, WEAPON)
		end
		gui.text(XTEXT,140,string.format("Random: %.2f %%", 100*tonumber(random_rate)),"white")
		gui.text(XTEXT,160,string.format("Current reward : %.2f", total_reward),"white")
		--gui.text(XTEXT,180,string.format("Average reward : %.2f", average_reward),"white")
		--gui.text(XTEXT,200,string.format("Best reward : %.2f", best_reward),"white")
		gui.text(XTEXT,180,"Episode : " .. n_episode,"white")
		if (MODE == 'Cup') or (MODE == 'Race') then
			gui.text(XTEXT,200,string.format("Average finish : %.2f", average_finish),"white")
		end
		
		gui.text(XTEXT,240,tracks[TRACK+1] .. " stats :","yellow")
		gui.text(XTEXT,260,"Finished attempts : " .. finished_attempts_track[TRACK+1],"yellow")
		gui.text(XTEXT,280,string.format("Average time : %d", average_time_track[TRACK+1]),"yellow")
		gui.text(XTEXT,300,string.format("Best time : %d", best_time_track[TRACK+1]),"yellow")
		if (MODE == 'Cup') or (MODE == 'Race') then
			gui.text(XTEXT,320,string.format("Average finish : %.2f", average_finish_track[TRACK+1]),"yellow")
			gui.text(XTEXT,340,string.format("Best finish : %d", best_finish_track[TRACK+1]),"yellow")
		end
	end 
	emu.frameadvance()
	
end
