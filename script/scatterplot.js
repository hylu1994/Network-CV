import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

const mapValue = (data, a) =>
  typeof a === 'function' ? Object.assign(d3.map(data, a), {
    type: 'function'
  }) : // mapping from data to size
    Array.isArray(a) ? Object.assign([...a], {
      type: 'array'
    }) : // array
      Object.assign(data.map(() => a), {
        type: 'constant'
      }); // constant number

export const scatterplot = (data, {
  svgId = 'scatterplot',
  x = ([x]) => x, // given d in data, returns the (quantitative) x-value
  y = ([, y]) => y, // given d in data, returns the (quantitative) y-value
  order, // order of points rendered
  links = [],
  r = 10,
  c = '#4D7AA7',
  stroke = '#CCCCCC', // stroke color for the dots
  strokeWidth = 1, // stroke width for dots
  marginTop = 25, // top margin, in pixels
  marginRight = 30, // right margin, in pixels
  marginBottom = 40, // bottom margin, in pixels
  marginLeft = 40, // left margin, in pixels
  width = 640, // outer width, in pixels
  height = width, // outer height, in pixels
  xType = d3.scaleLinear, // type of x-scale
  xDomain, // [xmin, xmax]
  xRange, // [left, right]
  yType = d3.scaleLinear, // type of y-scale
  yDomain, // [ymin, ymax]
  yRange, // [bottom, top]
  xLabel, // a label for the x-axis
  yLabel, // a label for the y-axis
  showColorLegend = 'auto',
  linkStroke = '#DDDDDD', // link's line color
  linkStrokeWidth = 1, // link's line Width
  linkStrokeOpacity = 0.7, // link's opacity
  showXAxis = true,
  showYAxis = true,
  xTicksCount,
  yTicksCount
} = {}) => {
  // Compute values.
  const X = mapValue(data, x);
  const Y = mapValue(data, y);
  const C = mapValue(data, c);
  const I = order ? order : d3.range(X.length);

  // Compute default domains.
  if (xDomain === undefined) xDomain = d3.extent(X);
  if (yDomain === undefined) yDomain = d3.extent(Y);

  // Construct scales and axes.
  if (xRange === undefined) xRange = [marginLeft, width - marginRight];
  if (yRange === undefined) yRange = [height - marginBottom, marginTop];

  const xScale = xType(xDomain, xRange);
  const yScale = yType(yDomain, yRange);

  const xAxis = d3.axisBottom(xScale).ticks(xTicksCount ? xTicksCount : width / 80).tickSizeOuter(0);
  const yAxis = d3.axisLeft(yScale).ticks(yTicksCount ? yTicksCount : height / 80).tickSizeOuter(0);

  // prepare SVG
  const svg = d3.create('svg')
    .attr('id', svgId)
    .attr('width', width)
    .attr('height', height)
    .attr('viewBox', [0, 0, width, height])
    .attr('style', 'max-width: 100%; height: auto; height: intrinsic;');

  // draw x, y-axes
  if (showXAxis) {
    const xAxisG = svg.append('g')
      .attr('id', `${svgId}-xaxis`)
      .attr('class', 'axis')
      .attr('transform', `translate(0,${height - marginBottom})`)
      .call(xAxis)
      .call(g => g.append('text')
        .attr('x', marginLeft + (width - marginLeft - marginRight) / 2)
        .attr('y', marginBottom - 4)
        .attr('fill', 'currentColor')
        .attr('text-anchor', 'center')
        .text(xLabel));
  }
  if (showYAxis) {
    const yAxisG = svg.append('g')
      .attr('id', `${svgId}-yaxis`)
      .attr('class', 'axis')
      .attr('transform', `translate(${marginLeft},0)`)
      .call(yAxis)
      // .call(g => g.append('text')
      //   .attr('x', -marginLeft + 2)
      //   .attr('y', 10)
      //   .attr('fill', 'currentColor')
      //   .attr('text-anchor', 'start')
      //   .text(yLabel));
      .call(g => g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -height / 2)
        .attr('y', -marginLeft + 10)
        .attr('fill', 'currentColor')
        .attr('text-anchor', 'middle')
        .text(yLabel));
  }
  // draw links in SVG if exists
  if (links) {
    svg.append('g')
      .attr('stroke', linkStroke)
      .attr('stroke-width', linkStrokeWidth)
      .attr('stroke-opacity', linkStrokeOpacity)
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('x1', link => xScale(X[link[0]]))
      .attr('x2', link => xScale(X[link[1]]))
      .attr('y1', link => yScale(Y[link[0]]))
      .attr('y2', link => yScale(Y[link[1]]));
  }
  // draw points in SVG
  svg.append('g')
    .attr('stroke', stroke)
    .attr('stroke-width', strokeWidth)
    .selectAll('circle')
    .data(I)
    .join('circle')
    .attr('class', 'default')
    .attr('fill', i => C[i])
    .attr('cx', i => xScale(X[i]))
    .attr('cy', i => yScale(Y[i]))
    .attr('r', r)
    .attr('z-index', 0); // This will work after SVG 2 is released

  const update = (data, {
    mode = 'default',
    selected = [],
    selectedColor = '#FAE727',
    showOnlySelected = false
  } = {}) => {
    if (mode === 'default') {
      svg.selectAll('circle.default')
        .data(I)
        .attr('fill', i => C[i])
        .attr('opacity', 1)
        .attr('z-index', 0); // This will work after SVG 2 is released
      if (links) {
        svg.selectAll('line')
          .data(links)
          .attr('stroke-opacity', 1);
      }
      // after SVG 2 is released, the code below should be removed for the performance
      svg.selectAll('circle.decorative').remove();
    } else {
      const orderedSelected = !order ? selected : order.map(idx => selected[idx]);
      svg.selectAll('circle.default')
        .data(I)
        .attr('fill', (i, idx) => orderedSelected[idx] ? (selectedColor ? selectedColor : C[i]) : C[i])
        .attr('z-index', (i, idx) => orderedSelected[idx] ? 1 : 0); // This will work after SVG 2 is released
      if (showOnlySelected) {
        svg.selectAll('circle.default')
          .attr('opacity', (i, idx) => orderedSelected[idx] ? 1 : 0);
        if (links) {
          svg.selectAll('line')
            .data(links)
            .attr('stroke-opacity', link => (selected[link[0]] && selected[link[1]]) ? 1 : 0);
        }
      }

      // after SVG 2 is released, the code below should be removed for the performance
      svg.selectAll('circle.decorative').remove();
      svg.append('g')
        .attr('stroke', '#888888')
        .attr('stroke-width', strokeWidth)
        .selectAll('circle')
        .data(I)
        .join('circle')
        .attr('class', 'decorative')
        .attr('fill', (i, idx) => orderedSelected[idx] ? (selectedColor ? selectedColor : C[i]) : C[i])
        .attr('opacity', (i, idx) => orderedSelected[idx] ? 1 : 0)
        .attr('cx', i => xScale(X[i]))
        .attr('cy', i => yScale(Y[i]))
        .attr('r', r);
    }
  }

  return Object.assign(svg.node(), {
    update,
    xScale,
    yScale,
    order,
    marginTop,
    marginRight,
    marginBottom,
    marginLeft
  });
}