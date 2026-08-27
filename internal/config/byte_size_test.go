package config

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseByteSize(t *testing.T) {
	for _, tt := range []struct {
		input string
		want  ByteSize
	}{
		{"0B", 0},
		{"512B", 512},
		{"2KiB", 2 << 10},
		{"28GiB", 28 << 30},
		{"1TiB", 1 << 40},
		{" 1MiB ", 1 << 20},
	} {
		t.Run(tt.input, func(t *testing.T) {
			got, err := ParseByteSize(tt.input)
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestParseByteSize_RejectsAmbiguousAndOverflowingValues(t *testing.T) {
	for _, input := range []string{"", "1", "1GB", "1.5GiB", "-1GiB", "1 GiB", "KiB", "16777216TiB"} {
		t.Run(input, func(t *testing.T) {
			_, err := ParseByteSize(input)
			assert.Error(t, err)
		})
	}

	_, err := ParseByteSize("16777215TiB")
	require.NoError(t, err)

	got, err := ParseByteSize("18446744073709551615B")
	require.NoError(t, err)
	assert.Equal(t, ByteSize(math.MaxUint64), got)
}
