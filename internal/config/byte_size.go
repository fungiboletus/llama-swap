package config

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// ByteSize is a configured size represented internally as an exact byte count.
// YAML values must use one of B, KiB, MiB, GiB, or TiB so units are explicit.
type ByteSize uint64

// ParseByteSize parses an unsigned integer followed immediately by an IEC byte
// unit. Decimal values and unitless values are deliberately rejected.
func ParseByteSize(value string) (ByteSize, error) {
	value = strings.TrimSpace(value)
	for _, unit := range []struct {
		suffix     string
		multiplier uint64
	}{
		{"TiB", 1 << 40},
		{"GiB", 1 << 30},
		{"MiB", 1 << 20},
		{"KiB", 1 << 10},
		{"B", 1},
	} {
		if !strings.HasSuffix(value, unit.suffix) {
			continue
		}

		number := strings.TrimSuffix(value, unit.suffix)
		if number == "" {
			return 0, fmt.Errorf("size %q is missing a number", value)
		}
		parsed, err := strconv.ParseUint(number, 10, 64)
		if err != nil {
			return 0, fmt.Errorf("invalid size %q: use an unsigned integer followed by B, KiB, MiB, GiB, or TiB", value)
		}
		if parsed > math.MaxUint64/unit.multiplier {
			return 0, fmt.Errorf("size %q overflows bytes", value)
		}
		return ByteSize(parsed * unit.multiplier), nil
	}

	return 0, fmt.Errorf("invalid size %q: use an unsigned integer followed by B, KiB, MiB, GiB, or TiB", value)
}

// UnmarshalYAML decodes a size from its required string form.
func (s *ByteSize) UnmarshalYAML(value *yaml.Node) error {
	if value.Kind != yaml.ScalarNode || value.Tag != "!!str" {
		return fmt.Errorf("size must be a string using B, KiB, MiB, GiB, or TiB")
	}
	parsed, err := ParseByteSize(value.Value)
	if err != nil {
		return err
	}
	*s = parsed
	return nil
}
