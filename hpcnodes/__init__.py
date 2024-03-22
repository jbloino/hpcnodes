#!/usr/bin/env python3
"""HPC cluster node definition.

Provides nodes information as object to be used by other programs for HPC
    applications.

Attributes
----------
PATH_INIFILE : str
    Default file (and path) to hpcnodes.ini
BIT_POWERS : list
    List of symbols for the power of bytes (only expected for internal use)

Methods
-------
bytes_units
    Returns a storage specification from a number of bytes
convert_storage
    Returns the number of bytes in a given storage specification
list_queues_nodes
    Returns the full list of queues over a list of nodes families
parse_ini
    Parses a node specification file and returns a list of nodes families
"""

import os
import typing
from math import floor
from configparser import ConfigParser

# ================
# Module Constants
# ================

PATH_INIFILE = os.path.join(os.getenv('HOME'), 'hpcnodes.ini')
BIT_POWERS = (' ', 'k', 'm', 'g', 't', 'p', 'e', 'z', 'y')

# ==============
# Module Classes
# ==============


class NodeFamily(object):
    """Represents a family of computing nodes.

    Represents a family of nodes, defined by a same hardware
    architecture and referred to by the same family name.

    Parameters
    ----------
    name : str
        Name of the family.
    num_nodes : int
        Number of nodes belonging to the family.
    num_cpus : int, optional
        Number of physical processors.
    num_cores : int, optional
        Number of cores / physical processors.
    size_mem : int, optional
        Available RAM (in byte).
    size_disk : int, optional
        Available local storage (in byte).
    proc_model : str, optional
        Model of the physical processors (free format).
    proc_arch : str, optional
        Architecture family of the processors.
    proc_maker : str, optional
        Maker of the processors.
    num_gpus : int, optional
        Number of GPUs per node.
    gpu_model : str, optional
        Model of the GPUs (free format).
    gpu_arch : str, optional
        Architecture family of the GPUs.
    gpu_maker : str, optional
        Maker of the GPUs.
    qname : str, optional
        Name of the family for the queue manager
    supported_queues : str, optional
        List of supported queues.
    path_tmpdir : str, optional
        Path to temporary directory.
    user_groups : list, optional
        List of user groups with access to the node.
    cpu_limit : dict, optional
        Hard and Soft limits on the number of processors
    mem_limit : dict, optional
        Hard and Soft limits on the memory
    enabled : dict, optional
        Family of nodes available for use.
    """

    def __init__(self,
                 name: str,
                 num_nodes: int,
                 num_cpus: int = 1,
                 num_cores: int = 1,
                 size_mem: int = 0,
                 size_disk: int = 0,
                 virt_core: bool = False,
                 proc_model: typing.Optional[str] = None,
                 proc_arch: typing.Optional[str] = None,
                 proc_maker: typing.Optional[str] = None,
                 num_gpus: int = 0,
                 gpu_model: typing.Optional[str] = None,
                 gpu_arch: typing.Optional[str] = None,
                 gpu_maker: typing.Optional[str] = None,
                 qname: typing.Optional[str] = None,
                 queues: typing.Optional[typing.List[str]] = None,
                 path_tmp: typing.Optional[str] = None,
                 user_grps: typing.Optional[typing.List[str]] = None,
                 cpu_lim: typing.Optional[
                        typing.Dict[str, typing.Optional[int]]] = None,
                 mem_lim: typing.Optional[
                        typing.Dict[str, typing.Optional[int]]] = None,
                 enabled: bool = True
                 ):
        """Class constructor."""
        self.name = name
        # __num_nodes could be connected to a a class variable to count
        #   instance but little sense for our need
        self.__num_nodes = num_nodes
        self.ncpus = num_cpus
        self.ncores = num_cores
        self.size_mem = size_mem
        self.size_disk = size_disk
        self.cpu_model = proc_model
        self.cpu_arch = proc_arch
        self.cpu_maker = proc_maker
        self.ngpus = num_gpus
        self.gpu_model = gpu_model
        self.gpu_arch = gpu_arch
        self.gpu_maker = gpu_maker
        self.__core_virtual = virt_core
        self.queue_name = qname
        self.supported_queues = queues
        self.path_tmpdir = path_tmp
        self.user_groups = user_grps
        self.cpu_limits = cpu_lim
        self.mem_limits = mem_lim
        self.enabled = enabled

    # ===================================
    #   Decorators to access attributes
    # ===================================
    @property
    def name(self) -> str:
        """Name for the node family."""
        return self.__family_name

    @name.setter
    def name(self, name: str) -> None:
        self.__family_name = name

    @property
    def ncpus(self) -> int:
        """Count physical CPUs."""
        return self.__num_cpus

    @ncpus.setter
    def ncpus(self, num: int) -> None:
        if num <= 0:
            raise ValueError('Positive number of CPUs expected!')
        self.__num_cpus = num

    @property
    def ncores(self) -> int:
        """Count physical cores/processor."""
        return self.__num_cores

    @ncores.setter
    def ncores(self, num: int) -> None:
        if num <= 0:
            raise ValueError('Positive number of cores expected!')
        self.__num_cores = num

    @property
    def size_mem(self) -> int:
        """Available RAM (in byte) per node."""
        return self.__size_mem

    @size_mem.setter
    def size_mem(self, num: int) -> None:
        if num < 0:
            raise ValueError('Positive RAM expected!')
        self.__size_mem = num

    @property
    def size_disk(self) -> int:
        """Disk storage (in byte) per node."""
        return self.__size_storage

    @size_disk.setter
    def size_disk(self, num: int) -> None:
        if num < 0:
            raise ValueError('Positive storage expected!')
        self.__size_storage = num

    @property
    def cpu_model(self) -> typing.Union[str, None]:
        """Model of the CPUs in the nodes."""
        return self.__cpu_model

    @cpu_model.setter
    def cpu_model(self, label: typing.Union[str, None]) -> None:
        self.__cpu_model = label

    @property
    def cpu_arch(self) -> typing.Union[str, None]:
        """Architecture of the CPUs in the nodes."""
        return self.__cpu_arch

    @cpu_arch.setter
    def cpu_arch(self, label: typing.Union[str, None]) -> None:
        if label is None:
            self.__cpu_arch = None
        else:
            self.__cpu_arch = label.replace(' ', '').replace('-', '').lower()

    @property
    def cpu_maker(self) -> typing.Union[str, None]:
        """Architecture of the CPUs in the nodes."""
        return self.__cpu_maker

    @cpu_maker.setter
    def cpu_maker(self, label: typing.Union[str, None]) -> None:
        if label not in (None, 'Intel', 'AMD'):
            raise ValueError('Unrecognized CPU maker')
        self.__cpu_maker = label

    @property
    def ngpus(self) -> int:
        """Count physical GPUs."""
        return self.__num_gpus

    @ngpus.setter
    def ngpus(self, num: int) -> None:
        if num < 0:
            raise ValueError('Positive number of GPUs expected!')
        self.__num_gpus = num

    @property
    def gpu_model(self) -> typing.Union[str, None]:
        """Model of the GPUs in the nodes."""
        return self.__gpu_model

    @gpu_model.setter
    def gpu_model(self, label: typing.Union[str, None]) -> None:
        self.__gpu_model = label

    @property
    def gpu_arch(self) -> typing.Union[str, None]:
        """Architecture of the GPUs in the nodes."""
        return self.__gpu_arch

    @gpu_arch.setter
    def gpu_arch(self, label: typing.Union[str, None]) -> None:
        if label is None:
            self.__gpu_arch = None
        else:
            self.__gpu_arch = label.replace(' ', '').replace('-', '').lower()

    @property
    def gpu_maker(self) -> typing.Union[str, None]:
        """Architecture of the GPUs in the nodes."""
        return self.__gpu_maker

    @gpu_maker.setter
    def gpu_maker(self, label: typing.Union[str, None]) -> None:
        if label not in (None, 'NVidia', 'AMD'):
            raise ValueError('Unrecognized GPU maker')
        self.__gpu_maker = label

    @property
    def queue_name(self) -> typing.Union[str, None]:
        """Name of the node family for the queue manager."""
        return self.__qname

    @queue_name.setter
    def queue_name(self, label: typing.Union[str, None]) -> None:
        self.__qname = label

    @property
    def supported_queues(self) -> typing.Union[typing.List[str], None]:
        """List of supported queues."""
        return self.__queues

    @supported_queues.setter
    def supported_queues(self,
                         queues: typing.Optional[typing.List[str]]) -> None:
        self.__queues = queues

    @property
    def path_tmpdir(self) -> str:
        """Path to temporary storage (for scratch usage).

        Note
        ----
        For paths dependent on username, the format specification
        `{username}` can be specified.
        """
        return self.__tmpdir

    @path_tmpdir.setter
    def path_tmpdir(self, label: typing.Union[str, None]) -> None:
        self.__tmpdir = label

    @property
    def user_groups(self) -> typing.Union[typing.List[str], None]:
        """List of authorized user groups."""
        return self.__usergroups

    @user_groups.setter
    def user_groups(self, label: typing.Union[str, None]) -> None:
        self.__usergroups = label

    @property
    def cpu_limits(self) -> typing.Dict[str, typing.Optional[int]]:
        """CPU limits."""
        return self.__cpu_limits

    @cpu_limits.setter
    def cpu_limits(self,
                   label: typing.Union[
                       typing.Dict[str, typing.Optional[int]],
                       None]) -> None:
        self.__cpu_limits = {'soft': None, 'hard': None}
        if label is not None:
            for key in label.keys():
                if key in ('soft', 'hard'):
                    if label[key] is not None:
                        try:
                            self.__cpu_limits[key] = int(label[key])
                            if self.__cpu_limits[key] < 0:
                                msg = 'Positive number of CPUs expected!'
                                raise ValueError(msg)
                        except ValueError as err:
                            msg = f'Wrong definition of CPU {key} limit'
                            raise ValueError(msg) from err
                else:
                    raise KeyError('Unrecognized type of CPU limit.')

    @property
    def mem_limits(self) -> typing.Dict[str, typing.Optional[int]]:
        """Memory limits."""
        return self.__mem_limits

    @mem_limits.setter
    def mem_limits(self,
                   label: typing.Union[
                       typing.Dict[str, typing.Optional[int]],
                       None]) -> None:
        self.__mem_limits = {'soft': None, 'hard': None}
        if label is not None:
            for key in label.keys():
                if key in ('soft', 'hard'):
                    if label[key] is not None:
                        try:
                            self.__mem_limits[key] = int(label[key])
                            if self.__mem_limits[key] < 0:
                                msg = 'Positive memory expected!'
                                raise ValueError(msg)
                        except ValueError as err:
                            msg = f'Wrong definition of memory {key} limit'
                            raise ValueError(msg) from err
                else:
                    raise KeyError('Unrecognized type of mem limit.')

    @property
    def enabled(self) -> bool:
        """Family nodes are available."""
        return self.__enabled

    @enabled.setter
    def enabled(self, stat: bool):
        self.__enabled = bool(stat)

    # ===========
    #   Methods
    # ===========

    def nprocs(self, count_all: bool = True) -> int:
        """Return the total number of processors.

        Returns the total number of physical (and virtual) processors
        as: ncpus x ncores

        Parameters
        ----------
        count_all
            Includes the number of virtual processors.

        Returns
        -------
        int
            Total number of physical (and virtual) processors
        """
        if count_all and self.__core_virtual:
            factor = 2
        else:
            factor = 1
        return factor*self.ncpus*self.ncores

    # ===========================
    #   Python built-in methods
    # ===========================

    def __len__(self):
        """Return the number of nodes in the family."""
        return self.__num_nodes

    def __str__(self):
        """Return the result for string conversion."""
        fmt_head = """Family: {name} ({nnodes} nodes)
    {nprocs} {ncores}-core processors{virt}
    {mem} of RAM / {disk} of available storage
"""
        txt_mem = bytes_units(self.size_mem)
        txt_disk = bytes_units(self.size_disk)
        if self.__core_virtual:
            txt_virt = ' (virtual cores enabled)'
        else:
            txt_virt = ''
        text = fmt_head.format(name=self.name, nnodes=self.__num_nodes,
                               nprocs=self.ncpus, ncores=self.ncores,
                               virt=txt_virt, mem=txt_mem, disk=txt_disk)
        if self.cpu_model is not None:
            txt_maker = self.cpu_maker is None and 'N/A' or self.cpu_maker
            txt_arch = self.cpu_arch is None and 'N/A' or self.cpu_arch
            fmt_cpu = """\
    CPU model: {} (maker: {}, family: {})
"""
            text += fmt_cpu.format(self.cpu_model, txt_maker, txt_arch)
        if self.ngpus > 0:
            text += f"""\
    {self.ngpus} GPUs present in each node.
"""
            if self.gpu_model is not None:
                txt_maker = 'N/A' if self.gpu_maker is None else self.gpu_maker
                txt_arch = 'N/A' if self.gpu_arch is None else self.gpu_arch
                fmt_gpu = """\
        GPU model: {} (maker: {}, family: {})
"""
                text += fmt_gpu.format(self.cpu_model, txt_maker, txt_arch)
        return text


# ================
# Module Functions
# ================


def convert_storage(label: str) -> int:
    """Convert storage string to number of bytes.

    Given a storage specification with unit, converts it to a number of
    bytes.

    Parameters
    ----------
    label : str
        Storage specification (ex: 32GB)

    Returns
    -------
    int
        Number of bytes corresponding to the storage specification.
    """
    _label = label.strip().replace(' ', '').lower()
    if _label.endswith('ib'):
        metric = 1024
        offset = -3
        magnitude = _label[offset]
        str_num = _label[:offset]
    elif _label.endswith('b'):
        metric = 1000
        offset = -2
        magnitude = _label[offset]
        if magnitude not in BIT_POWERS:
            magnitude = None
            offset = -1
        str_num = _label[:offset]
    else:
        magnitude = None
        str_num = _label
    try:
        value = int(str_num)
    except ValueError as err:
        raise ValueError('Unsupported storage format.') from err
    if magnitude is not None:
        if magnitude not in BIT_POWERS:
            raise ValueError('Unsupported byte unit.')
        unit = metric**(BIT_POWERS.index(magnitude))
    else:
        unit = 1
    return value*unit


def bytes_units(num_bytes: int, prec: int = 0,
                binary: bool = False,
                power: typing.Optional[str] = None) -> str:
    """Convert a number of bytes to a human readable unit.

    Given a number of bytes (integer), returns a string with the
    storage expressed in a human readable form or in a specific
    unit.

    Parameters
    ----------
    num_bytes : int
        Total number of bytes
    prec : int
        Precision. Number of decimal digits.
    binary : bool
        Uses binary metric instead of SI metric.
    power : str
        Select specific power of bytes (must be in `BIT_POWERS`).

    Returns
    -------
    str
        Storage in a specific or most adapted unit.
    """
    if binary:
        metric = 1024
        end_unit = 'iB'
    else:
        metric = 1000
        end_unit = 'B'
    if prec < 0:
        raise ValueError('Precision must be positive or null')
    fmt = f'{{:.{prec:d}f}}{{}}'
    if power is None:
        magnitude = len(BIT_POWERS)
        while magnitude > 0:
            value = num_bytes/metric**magnitude
            if value >= 1.:
                break
            magnitude -= 1
    else:
        try:
            magnitude = BIT_POWERS.index(power.lower()[0])
            value = num_bytes/metric**magnitude
        except ValueError as err:
            raise ValueError('Unknown byte power.') from err
    return fmt.format(floor(value), BIT_POWERS[magnitude].upper()+end_unit)


def parse_ini(fname: str = PATH_INIFILE,
              include_NA: bool = False) -> typing.Dict[str, NodeFamily]:
    """Parse a configuration file and builds a list of node families.

    Parses a configuration file (ini-like) and builds a list of node
    familiies.

    Parameters
    ----------
    fname : str
        Path to the configuration file
    include_NA : bool
        Include non-available machines (with enabled = False).

    Returns
    -------
    dict
        Dictionary of node families.
    """
    type_qtype_list = typing.List[typing.Union[str, int]]

    def parse_queue_types(queue_types: str,
                          base_list: typing.Optional[type_qtype_list] = None,
                          ) -> type_qtype_list:
        """Parse a string containing a string with a queue type.

        Parses a string and generates a list of types of queue for the
        queue format.

        Parameters
        ----------
        queue_types
            String to parse
        base_list
            Starting list of queue types
        """
        items = [item.strip() for item in queue_types.split(',')]
        if base_list is None:
            try:
                new_list = [int(item) for item in items]
            except ValueError:
                new_list = [item for item in items]
        elif len(base_list) > 0:
            func_conv = type(base_list[0])
            new_list = base_list[:]
            for item in items:
                if item.startswith('-'):
                    value = func_conv(item[1:])
                    try:
                        new_list.remove(value)
                    except ValueError:
                        pass
                else:
                    if item.startswith('+'):
                        value = func_conv(item[1:])
                    else:
                        value = func_conv(item)
                    if value not in new_list:
                        new_list.append(value)
        return new_list

    if not os.path.exists(fname):
        raise FileNotFoundError('Missing configuration file')

    config = ConfigParser()
    config.read(fname)

    # First look for the General section to extract possible patterns
    queue_pattern = None
    queue_types = None
    if 'general' in config.sections():
        sec_general = config['general']
        queue_pattern = sec_general.get('QueueFormat')
        optname = 'QueueType'
        if optname in sec_general:
            queue_types = parse_queue_types(sec_general.get(optname))
    if queue_pattern is None:
        queue_pattern = 'q{qtype:02d}{qname}'

    nodes_list = {}
    # Builds list of HPC nodes families
    # ---------------------------------
    for section in config.sections():
        # Look for section starting with "family"
        if section.startswith('family.') and section != 'family.hpcnodestest':
            sec_data = config[section]
            # Mandatory data
            if 'Name' not in sec_data:
                raise ValueError('Family name mandatory.')
            name = sec_data.get('Name')
            optname = 'NodeCount'
            if optname not in sec_data:
                raise ValueError('The number of nodes should be an integer.')
            try:
                nnodes = sec_data.getint(optname)
            except ValueError as err:
                raise ValueError('Number of nodes mandatory.') from err
            # Check if enabled machine
            if not include_NA:
                status = sec_data.getboolean('Enabled', fallback=True)
                if not status:  # Family not available, switch to next.
                    continue
            # Main resources: number of procs, cores, storage, RAM
            nprocs = sec_data.getint('CPUCount')
            ncores = sec_data.getint('CoreCount')
            virt_core = sec_data.getboolean('CoreLogical', fallback=False)
            for optname in ('RAM', 'Storage'):
                if optname in sec_data:
                    try:
                        value = convert_storage(sec_data.get(optname))
                    except ValueError as err:
                        msg = f'Unsupported definition of the {optname}.'
                        raise ValueError(msg) from err
                else:
                    value = 0
                if optname == 'RAM':
                    size_mem = value
                else:
                    size_disk = value
            # Processor data
            cpumodel = sec_data.get('CPUModel')
            cpumaker = sec_data.get('CPUMaker')
            cpuarch = sec_data.get('CPUArch')
            # GPU data
            optname = 'GPUCount'
            if optname in sec_data:
                try:
                    ngpus = sec_data.getint(optname)
                except ValueError as err:
                    raise ValueError('GPUCount should be a positive number.') \
                        from err
            else:
                ngpus = 0
            gpumodel = sec_data.get('GPUModel')
            gpumaker = sec_data.get('GPUMaker')
            gpuarch = sec_data.get('GPUArch')
            # Queues data
            qname = sec_data.get('QueueName', name.lower())
            optname = 'QueueType'
            if optname in sec_data:
                qtypes = parse_queue_types(sec_data.get(optname),
                                           queue_types)
            else:
                qtypes = queue_types
            queues = []
            if qtypes is not None:
                for qtype in qtypes:
                    qlabel = queue_pattern.format(qtype=qtype, qname=qname)
                    queues.append(qlabel)
            optname = 'QueueList'
            if optname in sec_data:
                for item in sec_data.get(optname).split(','):
                    if item not in queues:
                        queues.append(item)
            # Path available to store temporary files
            path_tmp = sec_data.get('PathTemp')
            # Authorized groups
            value = sec_data.get('UserGroups')
            if value is None:
                user_groups = None
            else:
                user_groups = [item.strip() for item in value.split(',')]
            # Soft/Hard limits
            mem_lim = {
                'soft': sec_data.get('MemSoftLimit'),
                'hard': sec_data.get('MemHardLimit')
                }
            for key, mem in mem_lim.items():
                if mem is not None:
                    try:
                        mem_lim[key] = convert_storage(mem)
                    except ValueError as err:
                        msg = f'Unsupported definition of the {key} memory ' \
                            + 'limit.'
                        raise ValueError(msg) from err
            cpu_lim = {
                'soft': sec_data.get('CPUSoftLimit'),
                'hard': sec_data.get('CPUHardLimit')
                }
            for key, cpu in cpu_lim.items():
                if cpu is not None:
                    try:
                        cpu_lim[key] = convert_storage(cpu)
                    except ValueError as err:
                        msg = f'Unsupported definition of the {key} CPU limit.'
                        raise ValueError(msg) from err
            # Build node object
            nodes_list[name] = \
                NodeFamily(name, nnodes, nprocs, ncores, size_mem, size_disk,
                           virt_core, cpumodel, cpuarch, cpumaker, ngpus,
                           gpumodel, gpuarch, gpumaker, qname, queues,
                           path_tmp, user_groups, cpu_lim, mem_lim)

    return nodes_list


def list_queues_nodes(nodes_list: typing.Dict[str, NodeFamily]
                      ) -> typing.Dict[str, str]:
    """Return the full list of queues over a list of nodes.

    Given a list of node family, returns a dictionary with the full list of
    queues as keys and the node family name as value.

    Parameters
    ----------
    nodes_list : list
        List of :obj:`NodeFamily` objects.

    Returns
    -------
    dict
        list of queues with the family name corresponding to each queue.
    """
    queues = {}
    for name in nodes_list:
        for queue in nodes_list[name].supported_queues:
            queues[queue] = name

    return queues


if __name__ == '__main__':
    nodes = parse_ini('hpcnodes.ini')
    print(list_queues_nodes(nodes))
