-- Pixel Gen: AI pixel art generation plugin for Aseprite
-- Communicates with a local Python backend via WebSocket.

local json = dofile(app.fs.joinPath(app.fs.filePath(debug.getinfo(1).source:sub(2)), "json.lua"))
local sprite_utils = dofile(app.fs.joinPath(app.fs.filePath(debug.getinfo(1).source:sub(2)), "sprite_utils.lua"))

local SERVER_URL = "ws://127.0.0.1:9847"

-- State
local ws = nil
local generating = false
local current_result = nil
local current_palette = nil
local dlg = nil

-- Palette presets (common Lospec palettes)
local PALETTE_PRESETS = {
    "From Active Sprite",
    "Auto (8 colors)",
    "Auto (16 colors)",
    "Auto (32 colors)",
}

local function connect()
    if ws then return true end

    ws = WebSocket{
        url = SERVER_URL,
        onreceive = function(mt, data)
            if mt == WebSocketMessageType.TEXT then
                local msg = json.decode(data)

                if msg.type == "progress" then
                    if dlg then
                        dlg:modify{
                            id = "status",
                            text = string.format("Generating... step %d/%d", msg.step, msg.total)
                        }
                    end

                elseif msg.type == "result" then
                    -- Next binary message will be the image data
                    current_palette = msg.palette
                    if dlg then
                        dlg:modify{ id = "status", text = "Received result, waiting for image data..." }
                    end

                elseif msg.type == "error" then
                    generating = false
                    if dlg then
                        dlg:modify{ id = "status", text = "Error: " .. (msg.message or "unknown") }
                    end
                    app.alert("Pixel Gen Error: " .. (msg.message or "unknown"))

                elseif msg.type == "cancelled" then
                    generating = false
                    if dlg then
                        dlg:modify{ id = "status", text = "Cancelled." }
                    end

                elseif msg.type == "pong" then
                    if dlg then
                        dlg:modify{ id = "status", text = "Connected to server." }
                    end
                end

            elseif mt == WebSocketMessageType.BINARY then
                -- This is the image pixel data
                current_result = data
                generating = false
                if dlg then
                    dlg:modify{ id = "status", text = "Done! Click 'Apply to Sprite' to use." }
                end

            elseif mt == WebSocketMessageType.OPEN then
                if dlg then
                    dlg:modify{ id = "status", text = "Connected to server." }
                end

            elseif mt == WebSocketMessageType.CLOSE then
                ws = nil
                if dlg then
                    dlg:modify{ id = "status", text = "Disconnected from server." }
                end
            end
        end,
        deflate = false,
        minreconnectwait = 2,
        maxreconnectwait = 10,
    }
    ws:connect()
    return true
end

local function disconnect()
    if ws then
        ws:close()
        ws = nil
    end
end

local function send_generate(opts)
    if not ws then
        if not connect() then
            app.alert("Cannot connect to Pixel Gen server at " .. SERVER_URL)
            return
        end
    end

    generating = true
    current_result = nil
    current_palette = nil

    local request = {
        action = "generate",
        prompt = opts.prompt or "pixel art sprite",
        guidance_scale = opts.guidance_scale or 5.0,
        steps = opts.steps or 35,
        seed = opts.seed or -1,
        dither_mode = opts.dither_mode,
        outline_cleanup = opts.outline_cleanup,
        num_colors = opts.num_colors or 16,
    }

    -- Palette
    if opts.palette_source == "From Active Sprite" then
        local pal = sprite_utils.get_active_palette()
        if pal then
            request.palette = pal
        end
    end

    ws:sendText(json.encode(request))

    if dlg then
        dlg:modify{ id = "status", text = "Generating..." }
    end
end

local function show_generate_dialog()
    if dlg then
        dlg:close()
    end

    dlg = Dialog{ title = "Pixel Gen" }

    dlg:entry{ id = "prompt", label = "Prompt:", text = "pixel art knight character, side view" }
    dlg:combobox{ id = "palette_source", label = "Palette:", option = "Auto (16 colors)",
                  options = PALETTE_PRESETS }
    dlg:slider{ id = "guidance", label = "Guidance:", min = 10, max = 150, value = 50 }
    dlg:slider{ id = "steps", label = "Steps:", min = 10, max = 100, value = 35 }
    dlg:number{ id = "seed", label = "Seed (-1=random):", text = "-1", decimals = 0 }

    dlg:separator{ text = "Post-Processing" }
    dlg:combobox{ id = "dither", label = "Dithering:", option = "None",
                  options = { "None", "Ordered (Bayer)", "Floyd-Steinberg" } }
    dlg:check{ id = "outline", label = "Outline cleanup:", selected = true }

    dlg:separator()
    dlg:label{ id = "status", text = "Ready. Start server first: python -m server.main" }

    dlg:button{ id = "connect_btn", text = "Connect", onclick = function()
        connect()
        -- Send ping to verify
        if ws then ws:sendText(json.encode({ action = "ping" })) end
    end }

    dlg:button{ id = "generate_btn", text = "Generate", onclick = function()
        if generating then return end

        local dither_map = {
            ["None"] = nil,
            ["Ordered (Bayer)"] = "ordered",
            ["Floyd-Steinberg"] = "floyd_steinberg",
        }
        local palette_source = dlg.data.palette_source
        local num_colors = 16
        if palette_source == "Auto (8 colors)" then num_colors = 8
        elseif palette_source == "Auto (32 colors)" then num_colors = 32
        end

        send_generate({
            prompt = dlg.data.prompt,
            palette_source = palette_source,
            guidance_scale = dlg.data.guidance / 10.0,
            steps = dlg.data.steps,
            seed = tonumber(dlg.data.seed) or -1,
            dither_mode = dither_map[dlg.data.dither],
            outline_cleanup = dlg.data.outline,
            num_colors = num_colors,
        })
    end }

    dlg:button{ id = "apply_btn", text = "Apply to Sprite", onclick = function()
        if current_result and current_palette then
            sprite_utils.apply_pixels(current_result, 128, 128, current_palette)
        else
            app.alert("No generated image to apply. Generate first!")
        end
    end }

    dlg:button{ id = "cancel_btn", text = "Cancel", onclick = function()
        if ws and generating then
            ws:sendText(json.encode({ action = "cancel" }))
        end
    end }

    dlg:show{ wait = false }
end

function init(plugin)
    plugin:newMenuGroup{
        id = "pixel_gen_menu",
        title = "Pixel Gen",
        group = "sprite_properties",
    }

    plugin:newCommand{
        id = "pixel_gen_generate",
        title = "Generate Sprite...",
        group = "pixel_gen_menu",
        onclick = function()
            show_generate_dialog()
        end,
    }
end

function exit(plugin)
    disconnect()
    if dlg then dlg:close() end
end
