-- Sprite manipulation utilities for pixel_gen Aseprite plugin.

local M = {}

function M.apply_pixels(pixel_data, width, height, palette_hex)
    -- Apply generated pixel data to the active sprite (or create new one).
    -- pixel_data: raw RGBA bytes string (4 bytes per pixel, row-major)
    -- width, height: image dimensions
    -- palette_hex: array of "#rrggbb" strings

    app.transaction("Pixel Gen: Apply Generated Sprite", function()
        local sprite = app.sprite
        if not sprite then
            sprite = Sprite(width, height, ColorMode.RGB)
        end

        -- Ensure sprite is the right size
        if sprite.width ~= width or sprite.height ~= height then
            sprite:resize(width, height)
        end

        local layer = app.layer
        if not layer then
            layer = sprite.layers[1]
        end

        local frame = app.frame
        if not frame then
            frame = sprite.frames[1]
        end

        local cel = sprite:newCel(layer, frame)
        local image = Image(width, height, ColorMode.RGB)

        for y = 0, height - 1 do
            for x = 0, width - 1 do
                local idx = (y * width + x) * 4
                local r = pixel_data:byte(idx + 1) or 0
                local g = pixel_data:byte(idx + 2) or 0
                local b = pixel_data:byte(idx + 3) or 0
                local a = pixel_data:byte(idx + 4) or 0
                image:drawPixel(x, y, app.pixelColor.rgba(r, g, b, a))
            end
        end

        cel.image = image

        -- Set palette
        if palette_hex and #palette_hex > 0 then
            local palette = sprite.palettes[1]
            palette:resize(#palette_hex)
            for i, hex in ipairs(palette_hex) do
                local r = tonumber(hex:sub(2, 3), 16) or 0
                local g = tonumber(hex:sub(4, 5), 16) or 0
                local b = tonumber(hex:sub(6, 7), 16) or 0
                palette:setColor(i - 1, Color{ r = r, g = g, b = b, a = 255 })
            end
        end
    end)

    app.refresh()
end

function M.get_active_palette()
    -- Extract the current sprite's palette as an array of "#rrggbb" strings.
    local sprite = app.sprite
    if not sprite then return nil end

    local palette = sprite.palettes[1]
    if not palette then return nil end

    local colors = {}
    for i = 0, #palette - 1 do
        local c = palette:getColor(i)
        table.insert(colors, string.format("#%02x%02x%02x", c.red, c.green, c.blue))
    end
    return colors
end

function M.get_active_image_bytes()
    -- Get the active cel's image as raw RGBA bytes.
    local cel = app.cel
    if not cel then return nil, 0, 0 end

    local image = cel.image
    local w, h = image.width, image.height
    local bytes = {}

    for y = 0, h - 1 do
        for x = 0, w - 1 do
            local px = image:getPixel(x, y)
            local r = app.pixelColor.rgbaR(px)
            local g = app.pixelColor.rgbaG(px)
            local b = app.pixelColor.rgbaB(px)
            local a = app.pixelColor.rgbaA(px)
            table.insert(bytes, string.char(r, g, b, a))
        end
    end

    return table.concat(bytes), w, h
end

return M
