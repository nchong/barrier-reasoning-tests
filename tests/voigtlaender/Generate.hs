module Main (
    main
) where

import System.Environment

data Three = Zero | One | Two

instance Show Three where
    show Zero = "0"
    show One  = "1"
    show Two  = "2"

op1 :: Three -> Three -> Three
x    `op1` Zero = x
Zero `op1` One  = One
x    `op1` y    = Two

op2 :: Three -> Three -> Three
x `op2` Zero = x
x `op2` One  = One
x `op2` Two = Two

g1End :: Int -> Int -> [Three]
g1End len pos
    = replicate pos Zero
      ++ replicate (len - pos) Two

g1 :: Int -> Int -> [[Three]]
g1 len pos
    = map start $ map (g1End (len - pos - 1)) [0..(len - pos - 1)]
    where start end = replicate pos Zero ++ [One] ++ end

g2 :: Int -> Int -> [Three]
g2 len pos
    = replicate pos Zero
      ++ [One, Two]
      ++ replicate (len - pos - 2) Zero

ghe :: Int -> [Three] -> String
ghe 0 _      = ""
ghe 1 [x]    = show x
ghe n (x:xs) = show x ++ ", " ++ ghe (n - 1) xs

ghp :: Int -> ([Three], [Three]) -> String
ghp len (input, output)
    = "  {\n"
      ++ "    .input  = { " ++ (ghe len input)  ++ " },\n"
      ++ "    .output = { " ++ (ghe len output) ++ " }\n"
      ++ "  }"

gh :: Int -> [([Three], [Three])] -> String
gh _   []     = "};\n\n"
gh len [x]    = (ghp len x) ++ "\n};\n\n"
gh len (x:xs) = (ghp len x) ++ ",\n" ++ endString
    where endString = gh len xs

generateHeaderBegin :: Int -> String
generateHeaderBegin len
    = "#ifndef TEST_VECTOR_H_\n"
      ++ "#define TEST_VECTOR_H_\n\n"
      ++ "#define VECTOR_LENGTH " ++ show len ++ "\n\n"
      ++ "typedef struct {\n"
      ++ "  char input[VECTOR_LENGTH];\n"
      ++ "  char output[VECTOR_LENGTH];\n"
      ++ "} io_type;\n\n"

generateHeaderOne :: Int -> String
generateHeaderOne len
    = "#define OP1_MEMBERS " ++ show (length one) ++ "\n\n"
      ++ "io_type op1[] = {\n"
      ++ oneString
    where elements  = concat $ map (g1 len) [0..(len - 1)]
          results   = map (scanl1 op1) elements
          one       = zip elements results
          oneString = gh len one

generateHeaderTwo :: Int -> String
generateHeaderTwo len
    = "#define OP2_MEMBERS " ++ show (length two) ++ "\n\n"
      ++ "io_type op2[] = {\n"
      ++ twoString
    where elements  = map (g2 len) [0..(len - 2)]
          results   = map (scanl1 op2) elements
          two       = zip elements results
          twoString = gh len two

generateHeaderEnd :: String
generateHeaderEnd
    = "#endif\n"

main :: IO ()
main = do
    args <- getArgs

    len <- case args of
        [x] -> return (read x)
        _   -> error $ "wrong input"

    putStr $ generateHeaderBegin len
    putStr $ generateHeaderOne len
    putStr $ generateHeaderTwo len
    putStr $ generateHeaderEnd
