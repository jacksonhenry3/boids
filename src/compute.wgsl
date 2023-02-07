struct Particle {
  pos : vec2<f32>, //offset 0
  vel : vec2<f32>, //offset 8
  charge : f32, //offset 16
  padding:f32,
}; 

struct SimParams {
  deltaT : f32,
  rule1Distance : f32,
  rule2Distance : f32,
  rule3Distance : f32,
  rule1Scale : f32,
  rule2Scale : f32,
  rule3Scale : f32,
};




@group(0) @binding(0) var<uniform> params : SimParams;


@group(0) @binding(1) var<storage, read> particlesSrc : array<Particle>;
@group(0) @binding(2) var<storage, read_write> particlesDst : array<Particle>;

@compute
@workgroup_size(32)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
  let total = arrayLength(&particlesSrc);
  let index = global_invocation_id.x;

  // Early out if we're out of bounds (shouldnt be needed, but is safer)
  // if (index >= total) {
  //   return;
  // }

  var vPos : vec2<f32> = particlesSrc[index].pos;
  var vVel : vec2<f32> = particlesSrc[index].vel;
  var q1 : f32 = particlesSrc[index].charge;

  



  var force = vec2<f32>(0.0, 0.0);
  var i : u32 = 0u;



  loop {
    if (i >= total) {
      break;
    }
    if (i == index) {
      continue;
    }

    let pos = particlesSrc[i].pos;
    let vel = particlesSrc[i].vel;
    let q2 = particlesSrc[i].charge;

    //add electrical repulsion between particles
    let sep = vPos - pos;
    var dist = sqrt(dot(sep, sep));

    if (dist > 0.1) {
      continue;
    }

    // if (dist > 0.3) {
    //   continue;

    // }

    // avoid division by zero
    if (dist < 0.03) {
      dist = 0.03;
    }

    force = force + q1*q2*0.02*(sep / (dist*dist*dist));

    continuing {
      i = i + 1u;
    }
  }
  // air resistance
  force = force - 0.2*vVel;

  // bounce off the top
  if (vPos.y > 1.0 && vVel.y > 0.0) {
    vVel.y = -vVel.y;
    vPos.y = 1.0;
  }

  // bounce off the bottom
  if (vPos.y < -1.0 && vVel.y < 0.0) {
    vVel.y = -vVel.y;
    vPos.y = -1.0;
  }

  // bounce off the left
  if (vPos.x < -1.0 && vVel.x < 0.0) {
    vVel.x = -vVel.x;
    vPos.x = -1.0;
  }

  // bounce off the right
  if (vPos.x > 1.0 && vVel.x > 0.0) {
    vVel.x = -vVel.x;
    vPos.x = 1.0;
  }
  





  
  vVel = vVel + force* params.deltaT;

  // kinematic update
  vPos += vVel * params.deltaT;

  
particlesDst[index] = Particle(vPos, vVel, q1, 0.0);

  // Write back


}
