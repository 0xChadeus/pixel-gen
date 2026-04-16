-- Minimal JSON encoder/decoder for Aseprite Lua scripts.
-- Public domain. Based on rxi/json.lua (MIT).

local json = {}

local encode

local escape_char_map = {
  ["\\"] = "\\\\", ["\""] = "\\\"", ["\b"] = "\\b",
  ["\f"] = "\\f", ["\n"] = "\\n", ["\r"] = "\\r", ["\t"] = "\\t",
}

local function escape_char(c)
  return escape_char_map[c] or string.format("\\u%04x", c:byte())
end

local function encode_nil() return "null" end
local function encode_string(val)
  return '"' .. val:gsub('[%z\1-\31\\"]', escape_char) .. '"'
end
local function encode_number(val)
  if val ~= val or val <= -math.huge or val >= math.huge then
    error("unexpected number value '" .. tostring(val) .. "'")
  end
  return string.format("%.14g", val)
end

local function encode_table(val, stack)
  local res = {}
  stack = stack or {}
  if stack[val] then error("circular reference") end
  stack[val] = true

  if rawget(val, 1) ~= nil or next(val) == nil then
    -- Array
    local n = 0
    for k in pairs(val) do
      if type(k) ~= "number" then goto obj end
      n = n + 1
    end
    for i = 1, n do
      res[i] = encode(val[i], stack)
    end
    stack[val] = nil
    return "[" .. table.concat(res, ",") .. "]"
  end

  ::obj::
  local n = 0
  for k, v in pairs(val) do
    n = n + 1
    res[n] = encode_string(tostring(k)) .. ":" .. encode(v, stack)
  end
  stack[val] = nil
  return "{" .. table.concat(res, ",") .. "}"
end

encode = function(val, stack)
  local t = type(val)
  if t == "table" then return encode_table(val, stack)
  elseif t == "string" then return encode_string(val)
  elseif t == "number" then return encode_number(val)
  elseif t == "boolean" then return tostring(val)
  elseif t == "nil" then return "null"
  else error("unexpected type '" .. t .. "'") end
end

json.encode = function(val) return encode(val) end

-- Decoder
local function create_set(...)
  local s = {}
  for i = 1, select("#", ...) do s[select(i, ...)] = true end
  return s
end

local space_chars = create_set(" ", "\t", "\r", "\n")
local delim_chars = create_set(" ", "\t", "\r", "\n", "]", "}", ",")
local escape_chars = create_set("\\", "/", '"', "b", "f", "n", "r", "t", "u")
local literals = { ["true"] = true, ["false"] = false, ["null"] = nil }
local literal_map = { ['"'] = '"', ["\\"] = "\\", ["/"] = "/",
  ["b"] = "\b", ["f"] = "\f", ["n"] = "\n", ["r"] = "\r", ["t"] = "\t" }

local function next_char(str, idx, set, negate)
  for i = idx, #str do
    if set[str:sub(i, i)] ~= negate then return i end
  end
  return #str + 1
end

local function decode_error(str, idx, msg)
  local line_count = 1
  local col_count = 1
  for i = 1, idx - 1 do
    col_count = col_count + 1
    if str:sub(i, i) == "\n" then line_count = line_count + 1; col_count = 1 end
  end
  error(string.format("%s at line %d col %d", msg, line_count, col_count))
end

local function codepoint_to_utf8(n)
  if n <= 0x7f then return string.char(n)
  elseif n <= 0x7ff then return string.char(0xc0 + math.floor(n / 64), 0x80 + (n % 64))
  elseif n <= 0xffff then return string.char(0xe0 + math.floor(n / 4096), 0x80 + (math.floor(n / 64) % 64), 0x80 + (n % 64))
  else error("invalid codepoint") end
end

local function parse_unicode_escape(s)
  local n1 = tonumber(s:sub(1, 4), 16)
  local n2 = tonumber(s:sub(7, 10), 16)
  if n2 then n1 = (n1 - 0xd800) * 0x400 + (n2 - 0xdc00) + 0x10000 end
  return codepoint_to_utf8(n1)
end

local function parse_string(str, i)
  local res = {}
  local j = i + 1
  local k = j
  while j <= #str do
    local x = str:byte(j)
    if x < 32 then decode_error(str, j, "control character in string") end
    if x == 92 then -- backslash
      res[#res + 1] = str:sub(k, j - 1)
      j = j + 1
      local c = str:sub(j, j)
      if c == "u" then
        local hex = str:match("^[dD][89aAbB]%x%x\\u%x%x%x%x", j + 1) or str:match("^%x%x%x%x", j + 1)
        if not hex then decode_error(str, j + 1, "invalid unicode escape") end
        res[#res + 1] = parse_unicode_escape(hex)
        j = j + #hex
      else
        if not escape_chars[c] then decode_error(str, j, "invalid escape char '" .. c .. "'") end
        res[#res + 1] = literal_map[c]
      end
      k = j + 1
    elseif x == 34 then -- quote
      res[#res + 1] = str:sub(k, j - 1)
      return table.concat(res), j + 1
    end
    j = j + 1
  end
  decode_error(str, i, "expected closing quote")
end

local function parse_number(str, i)
  local x = next_char(str, i, delim_chars)
  local s = str:sub(i, x - 1)
  local n = tonumber(s)
  if not n then decode_error(str, i, "invalid number '" .. s .. "'") end
  return n, x
end

local function parse_literal(str, i)
  local x = next_char(str, i, delim_chars)
  local word = str:sub(i, x - 1)
  if literals[word] == nil and word ~= "null" then
    decode_error(str, i, "invalid literal '" .. word .. "'")
  end
  return literals[word], x
end

local parse

local function parse_array(str, i)
  local res = {}
  local n = 0
  i = i + 1
  while true do
    local x
    i = next_char(str, i, space_chars, true)
    if str:sub(i, i) == "]" then return res, i + 1 end
    x, i = parse(str, i)
    n = n + 1; res[n] = x
    i = next_char(str, i, space_chars, true)
    local c = str:sub(i, i)
    if c == "]" then return res, i + 1 end
    if c ~= "," then decode_error(str, i, "expected ']' or ','") end
    i = i + 1
  end
end

local function parse_object(str, i)
  local res = {}
  i = i + 1
  while true do
    local key, val
    i = next_char(str, i, space_chars, true)
    if str:sub(i, i) == "}" then return res, i + 1 end
    if str:sub(i, i) ~= '"' then decode_error(str, i, "expected string for key") end
    key, i = parse_string(str, i)
    i = next_char(str, i, space_chars, true)
    if str:sub(i, i) ~= ":" then decode_error(str, i, "expected ':'") end
    val, i = parse(str, next_char(str, i + 1, space_chars, true))
    res[key] = val
    i = next_char(str, i, space_chars, true)
    local c = str:sub(i, i)
    if c == "}" then return res, i + 1 end
    if c ~= "," then decode_error(str, i, "expected '}' or ','") end
    i = i + 1
  end
end

local char_func_map = {
  ['"'] = parse_string, ["0"] = parse_number, ["1"] = parse_number,
  ["2"] = parse_number, ["3"] = parse_number, ["4"] = parse_number,
  ["5"] = parse_number, ["6"] = parse_number, ["7"] = parse_number,
  ["8"] = parse_number, ["9"] = parse_number, ["-"] = parse_number,
  ["t"] = parse_literal, ["f"] = parse_literal, ["n"] = parse_literal,
  ["["] = parse_array, ["{"] = parse_object,
}

parse = function(str, idx)
  local c = str:sub(idx, idx)
  local f = char_func_map[c]
  if f then return f(str, idx) end
  decode_error(str, idx, "unexpected character '" .. c .. "'")
end

json.decode = function(str)
  if type(str) ~= "string" then error("expected argument of type string, got " .. type(str)) end
  local res, idx = parse(str, next_char(str, 1, space_chars, true))
  idx = next_char(str, idx, space_chars, true)
  if idx <= #str then decode_error(str, idx, "trailing garbage") end
  return res
end

return json
