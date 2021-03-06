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
local SAVESTATESLOT = 0

local function press_buttons()
    joypad.set({Left=BUTTONS[1], Right=BUTTONS[2], L1=BUTTONS[3], R1=BUTTONS[4], Square=BUTTONS[5], Cross=BUTTONS[6], Up=BUTTONS[7], Down=BUTTONS[8], Start=BUTTONS[9]}, 1)
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
	if frame == 100 then BUTTONS = {0, 0, 0, 0, 0, 1, 0, 0, 0}
	elseif frame == 120 then BUTTONS = {0, 0, 0, 0, 0, 0, 1, 0, 0}
	elseif frame == 140 then BUTTONS = {0, 0, 0, 0, 0, 1, 0, 0, 0}
	elseif frame == 160 then BUTTONS = {0, 0, 0, 0, 0, 1, 0, 0, 0}
	elseif frame == 180 then BUTTONS = {0, 0, 0, 0, 0, 1, 0, 0, 0}
	else BUTTONS = {0, 0, 0, 0, 0, 0, 0, 0, 0}
	end
	press_buttons()
end

local function restart_race(frame)
	if frame == 1 then BUTTONS = {0, 0, 0, 0, 0, 0, 0, 0, 1}
	elseif frame == 5 then BUTTONS = {0, 0, 0, 0, 0, 0, 0, 1, 0}
	elseif frame == 15 then BUTTONS = {0, 0, 0, 0, 0, 1, 0, 0, 0}
	else BUTTONS = {0, 0, 0, 0, 0, 0, 0, 0, 0}
	end
	press_buttons()
end

local function load_savestate(SAVESTATESLOT)
	savestate.loadslot(SAVESTATESLOT)
end

local X_POS, Y_POS, Z_POS, X_SPD, Y_SPD, Z_SPD
local X_POS_2, Y_POS_2, X_POS_3, Y_POS_3
local JUMP, ANGLE, RNG
local X_ANC = 0
local Y_ANC = 0
local SPD_CALC = 0

local XTEXT = 2
local YTEXT = 60

local XBOX = 670
local YBOX = 55

local XBOX2 = 10
local YBOX2 = 350

local PREV_LAPPROG, LAPPROG, LAPCOUNTER, RACEENDED, TIMER, DRIVEBACKWARDS, ISBACKWARDS
local PICKUP, PREV_PICKUP, WALL
local FRAMESENDED = 0
local ENDSWITCH = 1
local rewards = {}
local best_reward = 0
local n_episode = 0
local average_size = 100
local average_reward = 0
local is_random = 0

local HASH = gameinfo.getromhash()
print(HASH)



while true do
	--memory.write_s8(0x8D2B4, 3)
	--POINTER = memory.read_u32_le( 0x08D674 ) - 0x80000000
	--POINTER = memory.read_u32_le( 0x097FFC ) - 0x80000000
	if (HASH=='68A605C9') then POINTER = memory.read_u32_le( 0x09900C ) - 0x80000000 end		--US Version
	if (HASH=='5C0D56EF') then POINTER = memory.read_u32_le( 0x0993CC ) - 0x80000000 + 0x4  end	--PAL Version
	if (HASH=='07FE354E') then POINTER = memory.read_u32_le( 0x09C4CC ) - 0x80000000 + 0x4  end	--JAP Version
	--POINTER = memory.read_u32_le( 0x09902C ) - 0x80000000
	--POINTER = memory.read_u32_le( 0x1FFE30 ) - 0x80000000
	
	if (POINTER > 0) then
		X_POS_3 = X_POS_2
		X_POS_2 = X_POS
		Y_POS_3 = Y_POS_2
		Y_POS_2 = Y_POS
		PREV_LAPPROG = LAPPROG
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
		LAPPROG = 		(memory.read_s16_le( POINTER + 0x488 ))
		LAPCOUNTER = 		(memory.read_u8( POINTER + 0x44 ))
		RACEENDED = 		(memory.read_s8( POINTER + 0x2CB ))
		TIMER = 		(memory.read_s32_le( POINTER + 0x514 ))
		DRIVEBACKWARDS =(memory.read_s16_le( POINTER + 0x490 ))
		ISBACKWARDS = Bit((memory.read_s32_le( POINTER + 0x2C8 )), 9)
		WALL = 			(memory.read_u16_le( POINTER + 0x50 ))
		PICKUP = 		(memory.read_s8( POINTER + 0x376  ))

		
		TOT_SPD = math.floor(math.sqrt(X_SPD*X_SPD+Y_SPD*Y_SPD))

		-- gui.text(XTEXT,60,"Angle : " .. ANGLE,"white")
		gui.text(XTEXT,80,"Speed (RAM) : " .. RAM_SPD,"white")
		gui.text(XTEXT,100,"Speed (True): " .. TOT_SPD,"white")
		--gui.text(XTEXT,120,"Reserve : " .. TURBO,"white")
		--gui.text(XTEXT,140,"Charge : " .. TURBO_CHARGE,"white")
		--gui.text(XTEXT,160,"Jump : " .. JUMP,"white")
		
		-- gui.text(XTEXT,200,"Charge : " .. string.format("%08X", POINTER + 0x3E2),"white")
		--gui.text(XTEXT,200,string.format("%08X", POINTER),"white")
		
		gui.text(XTEXT,240,"X : " .. X_POS,"white")
		gui.text(XTEXT,260,"Y : " .. Y_POS,"white")
		--gui.text(XTEXT,280,"Z : " .. Z_POS,"white")
		--gui.text(XTEXT,300,"Angle : " .. ANGLE,"white")
		gui.text(XTEXT,320,"Lap Progress : " .. LAPPROG,"white")
		gui.text(XTEXT,340,"Drive Backwards : " .. DRIVEBACKWARDS,"white")
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
		
		X_ANC = X_POS
		Y_ANC = Y_POS
		--if LAPCOUNTER == 1 then
		--	memory.write_s8( POINTER + 0x2CB, 2)
		--end

		table=joypad.getimmediate()
		action = 0
		if is_random == 0 then color = "cyan" else color = "darkred" end
		if table["P1 Left"] then 
			gui.drawBox(164,375,176,385,color,color)
			action = 1 
		else end
		if table["P1 Right"] then 
			gui.drawBox(188,375,200,385,color,color)
			action = 2
		else end
		if table["P1 Up"] then gui.drawBox(177,362,187,377,color,color) else end
		if table["P1 Down"] then gui.drawBox(177,383,187,398,color,color) else end
		if table["P1 Select"] then gui.drawBox(214,377,231,383,color,color) else end
		if table["P1 Start"] then gui.drawBox(261,377,274,383,color,color) else end
		if table["P1 Square"] then gui.drawBox(242,373,298,390,color,color) else end
		if table["P1 Circle"] then gui.drawBox(320,373,334,387,color,color) else end
		if table["P1 Triangle"] then gui.drawBox(302,356,316,370,color,color) else end
		if table["P1 Cross"] then gui.drawBox(300,391,316,406,color,color) else end
		if table["P1 L1"] then gui.drawBox(174,334,190,343,color,color) else end
		if table["P1 L2"] then gui.drawBox(173,314,191,327,color,color) else end
		if table["P1 R1"] then gui.drawBox(301,334,315,343,color,color) else end
		if table["P1 R2"] then gui.drawBox(300,315,320,329,color,color) else end
		gui.drawImage("C:/Users/Justin/Documents/CTR/BizHawk-2.4.1/Test2.png",146,314,200,142)
		if is_random == 2 and ENDSWITCH == 0 and PREV_PICKUP < 4 then --manual training
			tcp:send(action)
		end

		if (RACEENDED ~= 2) and (TIMER > 0) and ((DRIVEBACKWARDS < 1000) or (ISBACKWARDS == 0)) and (PREV_PICKUP < 4) then
			if ENDSWITCH == 1 then
				ENDSWITCH = 0
				FRAMESENDED = 0
				total_reward = 0
				n_episode = n_episode + 1
			end
			FRAMESENDED = FRAMESENDED + 1
			reward = (PREV_LAPPROG - LAPPROG) / 33.0 * RAM_SPD / 12000.0
			if math.abs(PREV_LAPPROG - LAPPROG) > 5000 then reward = 0 
			end
			if WALL > 0 then reward = reward - 0.1 end
			if PICKUP > 3 then reward = -50 end

			total_reward = total_reward + reward
			gui.text(XTEXT,380,string.format("Current reward : %.2f", total_reward),"white")
			gui.text(XTEXT,400,string.format("Average reward : %.2f", average_reward),"white")
			gui.text(XTEXT,420,string.format("Best reward : %.2f", best_reward),"white")
			gui.text(XTEXT,440,"Attempts : " .. n_episode,"white")
			
			if FRAMESENDED > 1 then
				tcp:send(reward)
				feedback, status, partial = tcp:receive()
			end
			send_str = ""
			send_str = X_POS .. " " .. Y_POS .. " " .. X_POS_2 .. " " .. Y_POS_2 .. " " .. X_POS_3 .. " " .. Y_POS_3
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
		    gui.text(XTEXT,360,string.format("Random: %.2f %%", 100*tonumber(random_rate)),"white")
		    if is_random ~= 2 then
		    	press_buttons()
		    else 
		    	joypad.set({Cross = 1}, 1)
		    end
		elseif (RACEENDED == 2) then
			if ENDSWITCH == 0 then
				ENDSWITCH = 1
				FRAMESENDED = 0
				tcp:send(5000) --signal to end the episode
				feedback, status, partial = tcp:receive()
				rewards[n_episode % average_size] = total_reward
				if total_reward > best_reward then
					best_reward = total_reward
				end
				average_reward = average(rewards)
			end
			FRAMESENDED = FRAMESENDED + 1
			load_savestate(1)
		elseif (DRIVEBACKWARDS >= 1000) and (ISBACKWARDS == 1) then --if drive backwards restart
			if ENDSWITCH == 0 then
				ENDSWITCH = 1
				FRAMESENDED = 0
				tcp:send(5000) --signal to end the episode
				feedback, status, partial = tcp:receive()
				rewards[n_episode % average_size] = total_reward
				if total_reward > best_reward then
					best_reward = total_reward
				end
				average_reward = average(rewards)
			end
			FRAMESENDED = FRAMESENDED + 1
			load_savestate(1)
		end

	end 
	emu.frameadvance()
	
end
