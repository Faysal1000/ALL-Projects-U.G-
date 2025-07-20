using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace LifeinnovirorMentalHealthConsultency.Context.Tables
{
    public class Specialization
    {

        [Key]
        public int SpecializationId { get; set; }

        [Required(ErrorMessage = "Specialization name is required.")]
        [StringLength(100, ErrorMessage = "Specialization name cannot exceed 100 characters.")]
        public string Name { get; set; }

    }
}