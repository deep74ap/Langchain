from langchain.text_splitter import RecursiveCharacterTextSplitter
from torch import chunk

text = '''CPU initializes itself and looks for a firmware program (BIOS) stored in
BIOS Chip (Basic input-output system chip is a ROM chip found on
mother board that allows to access & setup computer system at most
basic level.)
1. In modern PCs, CPU loads UEFI (Unified extensible firmware
interface)

iii. CPU runs the BIOS which tests and initializes system hardware. Bios
loads configuration settings. If something is not appropriate (like missing
RAM) error is thrown and boot process is stopped.
This is called POST (Power on self-test) process.
(UEFI can do a lot more than just initialize hardware; it’s really a tiny
operating system. For example, Intel CPUs have the Intel Management
Engine. This provides a variety of features, including powering Intel’s
Active Management'''

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0
)

res = splitter.split_text(text)
print(len(res))
print(res)