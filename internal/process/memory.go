package process

import (
	"context"
	"fmt"

	ps "github.com/shirou/gopsutil/v4/process"
)

// ResidentMemoryBytes reports the current resident-set size of the launched
// command and all of its descendants. Descendants matter for launchers that
// keep an API server in the root process and run the model in a child process.
func (p *ProcessCommand) ResidentMemoryBytes(ctx context.Context) (uint64, error) {
	pid := p.pid.Load()
	if pid <= 0 {
		return 0, fmt.Errorf("[%s] process is not ready", p.id)
	}
	return processTreeResidentBytes(ctx, int32(pid))
}

func processTreeResidentBytes(ctx context.Context, rootPID int32) (uint64, error) {
	root, err := ps.NewProcess(rootPID)
	if err != nil {
		return 0, fmt.Errorf("open process %d: %w", rootPID, err)
	}

	seen := make(map[int32]struct{})
	queue := []*ps.Process{root}
	var total uint64
	for len(queue) > 0 {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		current := queue[0]
		queue = queue[1:]
		if _, duplicate := seen[current.Pid]; duplicate {
			continue
		}
		seen[current.Pid] = struct{}{}

		info, err := current.MemoryInfoWithContext(ctx)
		if err != nil {
			if current.Pid == rootPID {
				return 0, fmt.Errorf("read process %d memory: %w", rootPID, err)
			}
			// A child may exit between tree discovery and inspection. Ignore it;
			// the next observation will see the current tree.
			continue
		}
		total += info.RSS

		children, err := current.ChildrenWithContext(ctx)
		if err == nil {
			queue = append(queue, children...)
		}
	}
	if total == 0 {
		return 0, fmt.Errorf("process tree %d reported no resident memory", rootPID)
	}
	return total, nil
}
