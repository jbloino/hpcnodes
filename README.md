# HPC Nodes Data Management


## Description
A Python3 library to parse a configuration file containing HPC nodes specifications.

The module provides primarily a class representing a family of computing nodes "NodeFamily".

It provides also conversion functions of storage units and a function to list all queues built on nodes.


## Structure of the configuration file
The configuration file can contain two types of information:

* The `[general]` block, which contains information on the pattern of the queue names and generic values applicable to all or most nodes.
* Family node blocks with titles of the form `family.name`, with `name` the name of the family.

### General block

Available fields are:
* `QueueFormat`: The general format of the queue system, using modern Python format specification (ex: `q{qtype:02d}{qname}`).  The system recognizes the following fields:

    - `qtype`: The type of queue
    - `qname`: The name of the queue.

* `QueueType`: List of types of queue, separated by commas (ex: `02, 07, 14, 28`).

### Nodes family specifications
The parser supports the following information for a given family node:

* `Name` - *str* - name of the nodes family.
* `NodeCount` - *int* - number of available nodes.
* `QueueName` - *str* - name of the family in the general queue format, passed to `qname`.
* `QueueType` - *str* - available types of queues, preceded by `+` or `-` to add types compared to the list provided in the "general" block, respectively
* `QueueList` - *str* - List of queues, overriding the general format.
* `CPUCount` - *int* - Number of sockets.
* `CoreCount` - *int* - Number of physical cores per socket
* `CoreLogical` - *bool* - If True/Yes, each physical core supports two hardware threads.
* `RAM` - *str* - Available quantity of RAM, with the unit included (ex: "256GB").
* `Storage` - *str* - Available physical storage for temporary files (ex: "1TB").
* `CPUModel` - *str* - Model of CPU, free format.
* `CPUMaker` - *str* - Maker of the CPU, free format (generally, "Intel", "AMD"...)
* `CPUArch` - *str* - CPU micro-architecture, in lowercase (ex: "sandybridge", "haswell", "zen2").  May be used by other tools to find the right installation.
* `GPUCount` - *int* - Number of available GPUs.
* `GPUModel` - *str* - Model of the GPUs, free format.
* `GPUMaker` - *str* - Make of the GPUs, free format (generally, "NVidia")
* `GPUArch` - *str* - GPU architecture. May be used by other tools to choose a suitable version of a program.
* `CPUSoftLimit` - *int* - Soft limit on the total number of usable cores.  This should be used by programs to set the default maximum limit but can be exceeded.
* `CPUHardLimit` - *int* - Hard limit on the number of cores.  Contrary to `CPUHardLimit`, this value should never be exceeded.
* `MemSoftLimit` - *str* - Soft limit on the total usable memory.
* `MemHardLimit` - *str* - Hard limit on the total usable memory.
* `PathTemp` - *str* - Path to store temporary files.
* `UserGroups` - *str* - User groups allowed to use the nodes.


### Example of INI file

    # The strings between {} follow Python's format specification

    [general]
    QueueFormat = q{qtype:02d}{qname}
    QueueType = 02, 07, 14, 28

    [family.hpcnodestest]
    Name = Test
    NodeCount = 8
    QueueName = test
    QueueType = +02, -07
    QueueList = q10test
    CPUCount = 2
    CoreCount = 6
    CoreLogical = False
    RAM = 6GB
    Storage = 2000GB
    CPUModel = Intel E5
    CPUMaker = Intel
    CPUArch = SandyBridge
    GPUCount = 1
    GPUModel = NVidia GeForce GTX
    GPUMaker = NVidia
    GPUArch = K20
    CPUSoftLimit = 16
    CPUHardLimit = 32
    MemSoftLimit = 2GB
    MemHardLimit = 4GB
    PathTemp = /local/scratch/{username}
    UserGroups = SGI

NOTE: The family `hpcnodestest` is recognized as a special case by the library system and ignored in the INI file.