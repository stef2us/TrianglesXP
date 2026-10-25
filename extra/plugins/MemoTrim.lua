-- ==========================================
-- MemoTrim - Helicopter Trim Save
-- ==========================================

local saved_pitch_trim = 0.0
local saved_roll_trim = 0.0

local debug_message = ""
local show_message_until = 0

local memory_loaded = false

local is_transitioning = false
local transition_start_time = 0
local initial_pitch_for_trans = 0.0
local initial_roll_for_trans = 0.0
local target_pitch_for_trans = 0.0
local target_roll_for_trans = 0.0
local transition_duration = 5.0
local transition_type = ""

DataRef("current_pitch_trim", "sim/cockpit2/controls/elevator_trim", "writable")
DataRef("current_roll_trim", "sim/cockpit2/controls/aileron_trim", "writable")
DataRef("sim_time", "sim/time/total_running_time_sec", "readonly")

function get_config_filepath()
    local acf_name = AIRCRAFT_FILENAME or "unknown_heli"
    return SCRIPT_DIRECTORY .. "memotrim_" .. acf_name .. ".txt"
end

function load_memotrim_file()
    local filepath = get_config_filepath()
    local file = io.open(filepath, "r")
    if file then
        local p = file:read("*n")
        local r = file:read("*n")
        file:close()
        if p and r then
            saved_pitch_trim = p
            saved_roll_trim = r
        end
    end
    memory_loaded = true
end

function save_heli_trim()
    if not memory_loaded then load_memotrim_file() end
    is_transitioning = false
    saved_pitch_trim = current_pitch_trim
    saved_roll_trim = current_roll_trim
    local filepath = get_config_filepath()
    local file = io.open(filepath, "w")
    if file then
        file:write(saved_pitch_trim .. "\n" .. saved_roll_trim)
        file:close()
    end
    debug_message = "MEMOTRIM : Trim SAVED"
    show_message_until = sim_time + 3 
end

function restore_heli_trim()
    if not memory_loaded then load_memotrim_file() end
    if is_transitioning then return end
    initial_pitch_for_trans = current_pitch_trim
    initial_roll_for_trans = current_roll_trim
    target_pitch_for_trans = saved_pitch_trim
    target_roll_for_trans = saved_roll_trim
    transition_start_time = sim_time
    is_transitioning = true
    transition_type = "RESTORE"
    debug_message = "MEMOTRIM : Trim RESTORING..."
    show_message_until = sim_time + transition_duration
end

function initiate_reset_trim()
    if is_transitioning then return end 
    initial_pitch_for_trans = current_pitch_trim
    initial_roll_for_trans = current_roll_trim
    target_pitch_for_trans = 0.0
    target_roll_for_trans = 0.0
    transition_start_time = sim_time
    is_transitioning = true
    transition_type = "RESET"
    debug_message = "MEMOTRIM : Trim RESETTING..."
    show_message_until = sim_time + transition_duration
end

function process_memotrim_logic()
    if is_transitioning then
        local elapsed = sim_time - transition_start_time
        if elapsed >= transition_duration then
            current_pitch_trim = target_pitch_for_trans
            current_roll_trim = target_roll_for_trans
            is_transitioning = false
            if transition_type == "RESET" then
                debug_message = "MEMOTRIM : Trim RESET DONE"
            else
                debug_message = "MEMOTRIM : Trim RESTORE DONE"
            end
            show_message_until = sim_time + 3
        else
            local progress = elapsed / transition_duration
            current_pitch_trim = initial_pitch_for_trans + (target_pitch_for_trans - initial_pitch_for_trans) * progress
            current_roll_trim = initial_roll_for_trans + (target_roll_for_trans - initial_roll_for_trans) * progress
        end
    end
end

function draw_memotrim_info()
    if not memory_loaded then load_memotrim_file() end
    if sim_time < show_message_until then
        draw_string(20, 60, debug_message, "green")
    end
end

do_every_frame("process_memotrim_logic()")
do_every_draw("draw_memotrim_info()")

create_command("FlyWithLua/MemoTrim/Save_Position", "Save Trim Position", "save_heli_trim()", "", "")
create_command("FlyWithLua/MemoTrim/Restore_Position", "Restore Trim Position", "restore_heli_trim()", "", "")
create_command("FlyWithLua/MemoTrim/Reset_Trim", "Reset Trim to Zero Gradually", "initiate_reset_trim()", "", "")