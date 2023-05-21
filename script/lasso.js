import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm"; // TODO: wanna avoid depending on d3

// TODO:
// 1. avoid depending on d3
// 2. remove svgArea argument (see below)
// 3. enable to define on events even in .call
// 4. make selection faster
// const lasso = lassoSelection();
// lasso.on('end', () => {
//   console.log(lasso.selected())
// });
// d3.select(chart).call(
//   lasso(chart, d3.select(chart).selectAll('circle').nodes()));


const context2d = (top, left, width, height) => {
  const canvas = document.createElement('canvas');
  canvas.width = width * devicePixelRatio;
  canvas.height = height * devicePixelRatio;
  canvas.style.width = width + 'px';
  canvas.style.position = 'absolute';
  canvas.style.top = `${top}px`;
  canvas.style.left = `${left}px`;

  const context = canvas.getContext('2d');
  context.scale(devicePixelRatio, devicePixelRatio);

  return context;
}

const isInsideLasso = (point, lasso) => {
  const x = point[0];
  const y = point[1];

  let isInside = false;
  let prev = lasso.at(-1);
  for (const curr of lasso) {
    if (((curr[1] < y && prev[1] >= y) ||
        (prev[1] < y && curr[1] >= y)) &&
      (curr[0] <= x || prev[0] <= x)) {

      isInside ^= (curr[0] + (y - curr[1]) / (prev[1] - curr[1]) * (prev[0] - curr[0])) < x;
    }
    prev = curr;
  }

  return isInside;
}

const insideLasso = (points, lasso) => {
  // use these range to reduce # of points checked
  // TODO: can do a smarter way e.g., using quad tree
  const lassoXs = lasso.map(pos => pos[0]);
  const lassoYs = lasso.map(pos => pos[1]);
  const minX = Math.min(...lassoXs);
  const maxX = Math.max(...lassoXs);
  const minY = Math.min(...lassoYs);
  const maxY = Math.max(...lassoYs);

  return points.map(point =>
    (point[0] >= minX && point[0] <= maxX) && (point[1] >= minY && point[1] <= maxY) ?
    isInsideLasso(point, lasso) :
    false);
}


const getCenters = svgItems => {
  const points = [];
  svgItems.forEach(item => {
    const rect = item.getBoundingClientRect();
    const x = rect.left + rect.width / 2;
    const y = rect.top + rect.height / 2;
    points.push([x, y]);
  });
  return points;
}

export const lassoSelection = ({
  stroke = 'black',
  strokeWidth = 1.5,
  on = {
    start: () => {},
    drag: () => {},
    end: () => {}
  }
} = {}) => {
  let selected = null;

  function selection(svgArea, svgItems, {
    marginTop = 0,
    marginRight = 0,
    marginBottom = 0,
    marginLeft = 0,
    order,
  } = {}) {
    const bcRect = svgArea.getBoundingClientRect();
    const width = bcRect.width;
    const height = bcRect.height;

    const context = context2d(
      bcRect.top + marginTop,
      bcRect.left + marginLeft,
      width - marginLeft - marginRight,
      height - marginTop - marginBottom);
    const curve = d3.curveBasisClosed(context);

    const points = getCenters(svgItems);

    const renderLasso = points => {
      context.clearRect(0, 0, width, height);
      context.strokeStyle = stroke;
      context.lineWidth = strokeWidth;

      context.beginPath();
      curve.lineStart();
      for (const point of points) {
        curve.point(...point);
      }
      if (points.length === 1) curve.point(...points[0]);
      curve.lineEnd();
      context.stroke();

      context.canvas.dispatchEvent(new CustomEvent('input'));
    }

    const lasso = () => {
      // relative: x, y positions used for drawing in canvas
      // absolute: x, y positions used for lasso selection of svg elements
      const stroke = {
        relative: [],
        absolute: []
      };

      return stroke;
    }

    const started = event => {
      d3.select('body').append(() => context.canvas);
      renderLasso([]);
      on.start(event);
    };
    const dragged = event => {
      event.subject.relative.push([event.x, event.y]);
      event.subject.absolute.push([event.sourceEvent.clientX, event.sourceEvent.clientY]);
      renderLasso(event.subject.relative);
      on.drag(event);
    }
    const ended = event => {
      selected = insideLasso(points, event.subject.absolute);
      if (order) {
        const tmpSelected = Array(selected.length).fill(false);
        for (let i = 0; i < selected.length; ++i) {
          tmpSelected[order[i]] = selected[i];
        }
        for (let i = 0; i < selected.length; ++i) {
          selected[i] = tmpSelected[i];
        }
      }

      d3.select(context.canvas).remove();
      on.end(event);
    };

    return d3.drag()
      .container(context.canvas)
      .subject(lasso)
      .on('start', started)
      .on('drag', dragged)
      .on('end', ended);
  }

  selection.on = function(eventType, eventFunc) {
    if (!arguments.length) return on;
    if (arguments.length === 1) return on[eventType];
    if (eventType in on) {
      on[eventType] = eventFunc;
    }
    return selection;
  };
  selection.selected = () => {
    return selected
  };

  return selection;
}