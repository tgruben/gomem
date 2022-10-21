package dataframe

import (
	"math/bits"

	"github.com/apache/arrow/go/v10/arrow"
)

type Resolver interface {
	Resolve(idx int) (int, int)
	NumRows() int
}

// returns
type ChunkResolver struct {
	offsets     []int
	cachedChunk int
	numRows     int
}

func NewChunkResolver(chunks *arrow.Column) ChunkResolver {
	numOffsets := len(chunks.Data().Chunks()) + 1
	offsets := make([]int, numOffsets, numOffsets)
	offset := 0
	for i, chunk := range chunks.Data().Chunks() {
		offset += chunk.Len()
		offsets[i+1] = offset
	}
	return ChunkResolver{offsets: offsets, numRows: offset}
}

func (cr *ChunkResolver) NumRows() int {
	return cr.numRows
}

func (cr *ChunkResolver) Resolve(idx int) (int, int) {
	if len(cr.offsets) <= 1 {
		return 0, idx
	}
	if idx >= cr.offsets[cr.cachedChunk] && idx < cr.offsets[cr.cachedChunk+1] {
		return cr.cachedChunk, idx - cr.offsets[cr.cachedChunk]
	}
	chunkIndex := cr.Bisect(idx)
	cr.cachedChunk = chunkIndex
	return chunkIndex, idx - cr.offsets[chunkIndex]
}

func (cr *ChunkResolver) Bisect(idx int) int {
	// Search [lo, lo + n)
	lo := 0
	n := len(cr.offsets)
	for n > 1 {
		m := n >> 1
		mid := lo + m
		if idx >= cr.offsets[mid] {
			lo = mid
			n -= m
		} else {
			n = m
		}
	}
	return lo
}

type IndexResolver struct {
	index      []uint32
	mask       uint32
	shardwidth uint32
}

// invariant this can never be bigger than shardwidth
func NewIndexResolver(size int, mask uint32) *IndexResolver {
	ir := &IndexResolver{}
	ir.index = make([]uint32, size, size)
	ir.mask = mask
	ir.shardwidth = uint32(bits.OnesCount64(uint64(mask)))
	return ir
}

func (ir *IndexResolver) Resolve(idx int) (int, int) {
	x := ir.index[idx]
	return int(x >> ir.shardwidth), int(x & ir.mask)
}

func (ir *IndexResolver) NumRows() int {
	return len(ir.index)
}

func (ir *IndexResolver) Set(i int, c int, offset int) {
	x := uint32((c << ir.shardwidth) | offset)
	ir.index[i] = x
}
