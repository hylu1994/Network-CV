// import * as d3 from 'https://cdn.skypack.dev/d3@7';
import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";
import {
  scatterplot
} from './scatterplot.js';
import {
  barChart
} from './barchart.js';
import {
  lassoSelection
} from './lasso.js';

const model = {
  shapCharts: undefined,
  compositeCharts: undefined,
  networkChart: undefined,
  shapData: undefined,
  shapXyDomain: undefined,
  compositeData: undefined,
  networkData: undefined,
  histData: undefined,
  classificationInfoData: undefined,
  labelData: undefined,
  orderData: undefined,
  labelColors: {
    '0': '#fb694a', // matplotlib.cm.Reds(0.5)
    '1': '#6aaed6', // matplotlib.cm.Blues(0.5)
    '-1': '#cfcfcf',
    'selected': '#fad727',
    'non-selected': '#bcebb5'
  },
  selected: undefined
};

const labelToName = label => {
  let name = label;
  if (name == 0) name = 'Class 0';
  else if (name == 1) name = 'Class 1';
  else if (name == -1) name = 'Non-labeled';
  else if (name == 'selected') name = 'Selected';
  else if (name == 'non-selected') name = 'Non-selected';

  return name;
}

const prepareLegend = () => {
  const width = d3.select('#aux-legend').node().getBoundingClientRect().width;
  const height = d3.select('#aux-legend').node().getBoundingClientRect().height;
  const svg = d3.select('#aux-legend').append('svg')
    .attr('id', 'legend')
    .attr('width', width)
    .attr('height', height)
    .attr('viewBox', [0, 0, width, height])
    .attr('style', 'max-width: 100%; height: auto; height: intrinsic;');

  svg.append('g').append('text')
    .attr('x', 0)
    .attr('y', 6)
    .attr('alignment-baseline', 'middle')
    .attr('fill', '#444444')
    .style('font-size', 12)
    .style('font-weight', 'bold')
    .text('Legend for scatterplots');
  svg.append('svg:image')
    .attr('x', 10)
    .attr('y', 16)
    .attr('width', 130)
    .attr('height', 96)
    .attr('xlink:href', 'image/polar_colormap.png');
}

const prepareSHAPPlot = (data, xyDomain) => {
  d3.select('#shap-plot').selectAll('*').remove();

  const height = d3.select('#shap-plot').node().getBoundingClientRect().height;

  const attrNames = Object.keys(data);
  const nAttrs = attrNames.length;

  const heightI = (height - 45) / nAttrs;
  const attrNameAreaWidth = 180;
  const charts = [];
  const lastAttrName = Object.keys(data).at(-1);
  for (const attrName in data) {
    const dataI = data[attrName];
    const orderI = dataI.map(d => d.order);
    const chart = scatterplot(dataI, {
      svgId: `shap-${attrName}`,
      x: d => d.x,
      y: d => d.y,
      c: d => d.c,
      order: orderI,
      xDomain: xyDomain.x,
      // yDomain: xyDomain.y,
      width: d3.select('#shap-plot').node().getBoundingClientRect().width,
      height: attrName === lastAttrName ? heightI + 25 : heightI,
      marginTop: 5,
      marginLeft: attrNameAreaWidth,
      marginBottom: attrName === lastAttrName ? 30 : 3,
      strokeWidth: 0.0,
      r: 2,
      xLabel: attrName === lastAttrName ? 'SHAP value' : null,
      showXAxis: true,
      showYAxis: true,
      xTicksCount: 3,
      yTicksCount: 2
    });

    d3.select(chart).append('line')
      .style('stroke', '#bbbbbb')
      .style('stroke-width', 1)
      .attr('x1', chart.xScale(0))
      .attr('x2', chart.xScale(0))
      .attr('y1', 0)
      .attr('y2', heightI);
    const attrNameRect = d3.select(chart).append('rect')
      .attr('x', 10)
      .attr('y', 5)
      .attr('rx', 5)
      .attr('ry', 5)
      .attr('height', heightI - 10)
      .attr('width', attrNameAreaWidth - 35)
      .style('fill', 'white')
      .style('cursor', 'pointer');
    const attrNameText = d3.select(chart).append('text')
      .attr('x', attrNameAreaWidth - 30)
      .attr('y', heightI / 2)
      .attr('alignment-baseline', 'middle')
      .attr('text-anchor', 'end')
      .style('cursor', 'pointer')
      .attr('fill', '#444444')
      .text(attrName);

    const mouseonEvent = (event, d) => {
      if (attrNameRect.attr('class') !== 'selected') {
        attrNameRect.attr('class', 'selected').style('fill', '#555555');
        attrNameText.attr('class', 'selected').style('fill', 'white');
      } else {
        attrNameRect.attr('class', null).style('fill', 'white');
        attrNameText.attr('class', null).style('fill', null);
      }
    }
    attrNameRect.on('click', mouseonEvent);
    attrNameText.on('click', mouseonEvent);

    charts.push(chart);
  }

  for (const chart of charts) {
    d3.select('#shap-plot').append(() => chart);
  }

  return charts;
}

// use closure to assign unique id
const genPrepareCompositePlot = () => {
  let counter = 0;
  const prepareCompositePlot = (data, info, {
    order
  } = {}) => {
    d3.select('#composite-plots').append('div')
      .attr('id', `composite-plot-${counter}`)
      .attr('class', 'composite-plot');

    const width = d3.select(`#composite-plot-${counter}`).node().getBoundingClientRect().width; // * 0.8;
    const height = d3.select(`#composite-plot-${counter}`).node().getBoundingClientRect().height;
    const chart = scatterplot(data, {
      svgId: 'composite',
      x: data.x,
      y: data.y,
      c: data.c,
      order: order,
      r: 4,
      strokeWidth: 0.1,
      stroke: '#AAAAAA',
      xLabel: 'LDA\'s 1D Representaion Value',
      yLabel: 'y',
      width: width,
      height: height,
      marginTop: 35,
      marginLeft: 40,
      marginBottom: 30,
      showYAxis: info && info.attr.length > 0
    });
    d3.select(`#composite-plot-${counter}`).append(() => chart);

    // append composite variable info
    if (info && info.attr.length > 0) {
      let varInfo = 'y =';
      for (let attrIdx = 0; attrIdx < info.attr.length; attrIdx++) {
        const w = parseFloat(info.weight[attrIdx]).toFixed(1);
        const weightStr = w >= 0 ? `+${w}` : `${w}`;
        varInfo += ` ${weightStr}(${info.attr[attrIdx]})`
      }
      const measure = info.correlationMeasure.charAt(0).toUpperCase() + info.correlationMeasure.slice(1);
      d3.select(chart).append('text')
        .attr('x', 10)
        .attr('y', 12)
        .attr('text-anchor', 'start')
        .style('font-size', 12)
        .style('fill', '#444444')
        .text(varInfo);
      d3.select(chart).append('text')
        .attr('x', 10)
        .attr('y', 27)
        .attr('text-anchor', 'start')
        .style('font-size', 12)
        .style('fill', '#444444')
        .text(`${measure}'s corr.: ${Number(info.correlation).toFixed(3)}, p-val: ${Number(info.pval).toExponential(2)}`);
    } else {
      d3.select(chart).append('text')
        .attr('x', 10)
        .attr('y', 12)
        .attr('text-anchor', 'start')
        .style('font-size', 12)
        .style('fill', '#444444')
        .text('Points are dodged along y-axis');
    }
    const id = counter;
    counter++;

    return Object.assign(chart, {
      id: id
    });
  }

  return prepareCompositePlot;
}
const prepareCompositePlot = genPrepareCompositePlot();

const prepareNetworkPlot = (nodes, links, {
  order
} = {}) => {
  d3.select('#network-plot').selectAll('*').remove();
  const nodeDraw = d3.map(nodes, d => [d.x, d.y]);
  const linkDraw = d3.map(links, d => [d.source, d.target]);
  const labels = d3.map(nodes, d => d.label);
  const colors = d3.map(labels, d => d > 0 ? model.labelColors[1] : model.labelColors[d]);
  const width = d3.select('#network-plot').node().getBoundingClientRect().width;
  const network = scatterplot(nodeDraw, {
    svgId: 'network',
    links: linkDraw,
    c: colors,
    order: order,
    width: width,
    height: d3.select('#network-plot').node().getBoundingClientRect().height,
    showXAxis: false,
    showYAxis: false,
    strokeWidth: 0.1,
    stroke: '#b0b0b0',
    linkStrokeWidth: 0.5,
    linkStrokeOpacity: 0.6,
    r: 2.5,
    marginTop: 15,
    marginRight: 20,
    marginBottom: 15,
    marginLeft: 20
  });
  d3.select('#network-plot').append(() => network);

  const networkSvg = d3.select('svg#network');
  networkSvg.append('g')
    .attr('transform', `translate(${width - 80}, 5)`)
    .attr('stroke', '#888888')
    .attr('stroke-width', 0.5)
    .selectAll('circle')
    .data([model.labelColors[0], model.labelColors[1], model.labelColors[-1]])
    .join('circle')
    .attr('fill', d => d)
    .attr('cx', 10)
    .attr('cy', (d, i) => 10 + i * 12)
    .attr('r', 3);
  networkSvg.append('g')
    .attr('transform', `translate(${width - 80}, 5)`)
    .selectAll('text')
    .data([0, 1, -1])
    .join('text')
    .attr('x', 20)
    .attr('y', (d, i) => 10 + i * 12)
    .attr('alignment-baseline', 'middle')
    .attr('fill', '#444444')
    .style('font-size', 10)
    .text(d => labelToName(d));

  return network;
}

const prepareHistPlot = (plotsCoord, selected, labels) => {
  d3.select('#plot-attr-hist').selectAll('*').remove();

  const colors = selected ? [model.labelColors['selected'], model.labelColors['non-selected']] : [model.labelColors[0], model.labelColors[1]]
  const width = d3.select('#plot-attr-hist').node().getBoundingClientRect().width;
  for (const [idx, plot] of plotsCoord.entries()) {
    const attrPlot = barChart(plot.x, {
      svgId: `attrPlot${idx}`,
      x: plot.x,
      y: plot.y,
      z: plot.z,
      colors: colors,
      xFormat: d => `${d.toFixed(1)}`,
      xLabel: plot.attr_name,
      yLabel: idx === 0 ? 'relative frequency' : null,
      width: width,
      height: d3.select('#plot-attr-hist').node().getBoundingClientRect().height / 5, // / 3,
      marginTop: 15,
      marginBottom: 30,
      marginLeft: 30,
      marginRight: 15,
    });
    d3.select('#plot-attr-hist').append(() => attrPlot)
  };

  const attrPlot0Svg = d3.select('svg#attrPlot0');
  attrPlot0Svg.append('g')
    .attr('transform', `translate(${width - 140}, 3)`)
    .attr('stroke', '#888888')
    .attr('stroke-width', 0.5)
    .selectAll('rect')
    .data(selected ? [model.labelColors['selected'], model.labelColors['non-selected']] : [model.labelColors[0], model.labelColors[1]])
    .join('rect')
    .attr('fill', d => d)
    .attr('x', (d, i) => i * 60)
    .attr('y', (d, i) => i * 0)
    .attr('width', 7)
    .attr('height', 7);
  attrPlot0Svg.append('g')
    .attr('transform', `translate(${width - 140}, 3)`)
    .selectAll('text')
    .data(selected ? ['selected', 'non-selected'] : [0, 1])
    .join('text')
    .attr('x', (d, i) => 10 + i * 60)
    .attr('y', (d, i) => 4 + i * 0)
    .attr('alignment-baseline', 'middle')
    .attr('fill', '#444444')
    .style('font-size', 10)
    .text(d => labelToName(d));
}

const prepareClassificationInfo = (data) => {
  d3.select('#target-variable-info').text(data.targetVariable);
  d3.select('#class0-info').text(data.class0);
  d3.select('#class1-info').text(data.class1);
  d3.select('#accuracy-info').text(data.accuracy);
}

const prepareLasso = (charts) => {
  const lasso = lassoSelection();
  lasso.on('end', () => {
    model.selected = lasso.selected(); // get list of selected/unselected in boolean array
    if (Math.max(...model.selected) === 0) { // nothing selected by lasso
      model.selected = null;
      for (const chart of charts) {
        chart.update(null, {
          mode: 'default'
        });
      }

      ws.send(JSON.stringify({
        action: messageActions.passData,
        content: {
          'type': 'hist',
          'selected': model.labelData.map(d => d > 0 ? 1 : 0),
          'topk': 5
        }
      }));
    } else {
      for (const chart of charts) {
        chart.update(null, {
          mode: 'selected',
          selected: !chart.order ?
            model.selected : chart.order.map(i => model.selected[i]),
          selectedColor: model.labelColors.selected
        });
      }

      ws.send(JSON.stringify({
        action: messageActions.passData,
        content: {
          'type': 'hist',
          'selected': model.selected,
          'topk': 5
        }
      }));
    }
  });

  for (const chart of charts) {
    d3.select(chart).call(
      lasso(chart, d3.select(chart).selectAll('circle.default').nodes(), {
        marginTop: chart.marginTop,
        marginRight: chart.marginRight,
        marginBottom: chart.marginBottom,
        marginLeft: chart.marginLeft,
        order: chart.order
      }));
  }
}

const prepareComposite = () => {
  const selectedAttrNames = d3.select('#shap-plot').selectAll('svg text.selected').nodes().map(d => d.innerHTML)
  const corrSelected = document.querySelector('input[name="opt-measure"]:checked').value;

  ws.send(JSON.stringify({
    action: messageActions.passData,
    content: {
      'type': 'composite',
      'selected_attr_names': selectedAttrNames,
      'target_correlation': corrSelected
    }
  }));
}

const websocketUrl = 'ws://localhost:9000';
const ws = new WebSocket(websocketUrl);
const messageActions = {
  passData: 0
};

prepareLegend();

ws.onopen = event => {
  document.getElementById('button-composite').addEventListener('click', prepareComposite, false);
  ws.send(JSON.stringify({
    action: messageActions.passData,
    content: {
      'type': 'shap',
    }
  }));
  ws.send(JSON.stringify({
    action: messageActions.passData,
    content: {
      'type': 'auxiliary',
    }
  }));
};

ws.onmessage = event => {
  const receivedData = JSON.parse(event.data);
  const type = receivedData.type;

  if (type === 'shap') {
    model.shapData = {}
    for (const attrName in receivedData.content) {
      model.shapData[attrName] = JSON.parse(receivedData.content[attrName]);
    }
    model.shapXyDomain = receivedData.xy_domain;
    model.labelData = receivedData.labels;
    model.shapCharts = prepareSHAPPlot(model.shapData, model.shapXyDomain);
    prepareLasso([...model.shapCharts]);
    prepareComposite();

    ws.send(JSON.stringify({
      action: messageActions.passData,
      content: {
        'type': 'hist',
        'selected': model.labelData.map(d => d > 0 ? 1 : 0),
        'topk': 5
      }
    }));
    ws.send(JSON.stringify({
      action: messageActions.passData,
      content: {
        'type': 'network'
      }
    }));

  } else if (receivedData.type === 'composite') {
    model.compositeData = JSON.parse(receivedData.content);
    model.orderData = receivedData.order;

    const chart = prepareCompositePlot(model.compositeData, {
      attr: receivedData.attr,
      weight: receivedData.weight,
      correlation: receivedData.correlation,
      correlationMeasure: receivedData.correlation_measure,
      pval: receivedData.pval,
    }, {
      order: model.orderData
    });

    if (!model.compositeCharts) {
      model.compositeCharts = {};
      model.compositeCharts[chart.id] = chart;
    } else {
      model.compositeCharts[chart.id] = chart;
    }

    d3.select(chart).append('text')
      .attr('x', d3.select(chart).node().getBoundingClientRect().width - 8)
      .attr('y', 13)
      .attr('text-anchor', 'middle')
      .attr('alignment-baseline', 'middle')
      .text('\u24E7') // x mark
      .on('click', () => {
        d3.select(chart.parentNode).remove();
        delete model.compositeCharts[chart.id];
        document.querySelector('#composite-plots').scrollTo({
          top: document.querySelector('#composite-plots').scrollHeight,
          left: 0,
          behavior: 'smooth'
        });
        prepareLasso([...model.shapCharts, ...Object.values(model.compositeCharts), model.networkChart]);
      });

    document.querySelector('#composite-plots').scrollTo({
      top: document.querySelector('#composite-plots').scrollHeight,
      left: 0,
      behavior: 'smooth'
    });

    if (model.networkChart) {
      prepareLasso([...model.shapCharts, ...Object.values(model.compositeCharts), model.networkChart]);
    } else {
      prepareLasso([...model.shapCharts, ...Object.values(model.compositeCharts)]);
    }

    document.querySelector('#composite-plots').addEventListener('scroll', event => {
      prepareLasso([...model.shapCharts, ...Object.values(model.compositeCharts), model.networkChart]);
    });
  } else if (type === 'network') {
    model.networkData = {
      nodes: receivedData.node,
      links: receivedData.link
    };
    model.networkChart = prepareNetworkPlot(model.networkData.nodes, model.networkData.links,
      { order: d3.map(receivedData.node, d => d.label).map((v, i) => [v, i]).sort().map(a => a[1]) }
    );

    if (model.compositeCharts) {
      prepareLasso([...model.shapCharts, ...Object.values(model.compositeCharts), model.networkChart]);
    } else {
      prepareLasso([...model.shapCharts, model.networkChart]);
    }
  } else if (type === 'hist') {
    model.histData = JSON.parse(receivedData.content);
    prepareHistPlot(model.histData, model.selected, model.labelData);
  } else if (type === 'auxiliary') {
    model.classificationInfoData = JSON.parse(receivedData.content);
    prepareClassificationInfo(model.classificationInfoData);
  }
};